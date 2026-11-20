//! Semantic compilation of latent-attention decoders (`DeepSeek` V2/V3 family).

use super::{CURRENT_SCHEMA_VERSION, activation, invalid, rotary};
use crate::{
    error::Result,
    layout::{
        AttentionLayerType, DecoderConfig, ExpertRouting, LatentDecoderConfig, LatentExpertsConfig,
        RotaryPairing,
    },
    semantic::{
        BlockLayoutSpec, DecoderLayerSpec, DecoderSpec, FeedForwardSpec, LatentAttentionSpec,
        MixerSpec, NormalizationSpec, PositionEncodingSpec, RotaryPairingSpec, RoutedExpertsSpec,
        RouterNormalization, SemanticModelSpec, SharedExpertSpec,
    },
};

pub(super) fn compile(
    decoder: &DecoderConfig,
    latent: &LatentDecoderConfig,
    norm: NormalizationSpec,
) -> Result<SemanticModelSpec> {
    let layers = (0..decoder.num_hidden_layers)
        .map(|index| {
            Ok(DecoderLayerSpec {
                index,
                input_norm: norm,
                post_attention_norm: norm,
                mixer: mixer(decoder, latent, index, norm)?,
                feed_forward: feed_forward(decoder, latent.experts.as_ref(), index)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SemanticModelSpec {
        schema_version: CURRENT_SCHEMA_VERSION,
        decoder: DecoderSpec {
            hidden_size: decoder.hidden_size,
            vocab_size: decoder.vocab_size,
            tie_word_embeddings: decoder.tie_word_embeddings,
            final_norm: norm,
            embedding_multiplier: decoder.embedding_multiplier.unwrap_or(1.0),
            layers,
            block_layout: BlockLayoutSpec::Unified,
            prediction_layers: Vec::new(),
        },
    })
}

fn mixer(
    decoder: &DecoderConfig,
    latent: &LatentDecoderConfig,
    index: usize,
    norm: NormalizationSpec,
) -> Result<MixerSpec> {
    if decoder.layer_type(index) != AttentionLayerType::Latent {
        return Err(invalid("latent decoders mixing other attention layers are not supported"));
    }
    let attention = &latent.attention;
    let position = match rotary(decoder, index) {
        // The rotary part always rotates completely; `partial_rotary_factor`
        // describes ordinary heads, not the latent's rope dimensions.
        PositionEncodingSpec::Rotary(rotary) => {
            PositionEncodingSpec::Rotary(crate::semantic::RotarySpec {
                partial_factor: 1.0,
                ..rotary
            })
        },
        PositionEncodingSpec::None => PositionEncodingSpec::None,
    };
    Ok(MixerSpec::LatentAttention(LatentAttentionSpec {
        query_heads: decoder.num_attention_heads,
        query_rank: attention.query_rank.map(std::num::NonZeroUsize::get),
        key_value_rank: attention.key_value_rank,
        nope_head_dim: attention.nope_head_dim,
        rope_head_dim: attention.rope_head_dim,
        value_head_dim: attention.value_head_dim,
        latent_normalization: norm,
        rotary_pairing: match attention.rotary_pairing {
            RotaryPairing::Interleaved => RotaryPairingSpec::Interleaved,
            RotaryPairing::Halves => RotaryPairingSpec::Halves,
        },
        scale: attention.scale,
        position,
    }))
}

fn feed_forward(
    decoder: &DecoderConfig,
    experts: Option<&LatentExpertsConfig>,
    index: usize,
) -> Result<FeedForwardSpec> {
    let activation = activation(decoder);
    let Some(experts) = experts.filter(|experts| index >= experts.first_dense_layers) else {
        return Ok(FeedForwardSpec::Dense {
            intermediate_size: decoder.intermediate_size,
            activation,
        });
    };
    let expert_count = decoder
        .num_experts
        .ok_or_else(|| invalid("routed latent decoder is missing its expert count"))?;
    let top_k = decoder
        .top_k_experts
        .ok_or_else(|| invalid("routed latent decoder is missing its top-k expert count"))?;
    let intermediate_size = decoder
        .moe_intermediate_size
        .ok_or_else(|| invalid("routed latent decoder is missing moe_intermediate_size"))?;
    let shared = experts
        .shared_experts
        .map(|count| {
            Ok::<_, crate::error::ModelsError>(SharedExpertSpec {
                intermediate_size: count
                    .get()
                    .checked_mul(intermediate_size)
                    .ok_or_else(|| invalid("shared expert width overflows"))?,
                activation: activation.clone(),
                gated_output: false,
            })
        })
        .transpose()?;
    Ok(FeedForwardSpec::Routed {
        routed: RoutedExpertsSpec {
            expert_count,
            top_k,
            intermediate_size,
            activation,
            router_normalization: match experts.routing {
                ExpertRouting::Softmax { renormalize: true } => RouterNormalization::SoftmaxTopK,
                ExpertRouting::Softmax { renormalize: false } => {
                    RouterNormalization::SoftmaxProbabilityTopK
                },
                ExpertRouting::SigmoidWithCorrectionBias => RouterNormalization::SigmoidBiasedTopK,
            },
            output_scale: experts.output_scale,
        },
        shared,
    })
}
