//! Semantic compilation of state-space hybrid decoders.

use super::{CURRENT_SCHEMA_VERSION, invalid};
use crate::{
    error::Result,
    layout::{
        DecoderConfig, HybridLayer, StateSpaceHybridConfig, StateSpaceStateDType, TimeStepBound,
    },
    semantic::{
        ActivationSpec, AttentionOutputSpec, AttentionSpec, BlockLayoutSpec, DecoderLayerSpec,
        DecoderSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, NormalizationSpec,
        PositionEncodingSpec, QkNormalization, RoutedExpertsSpec, RouterNormalization,
        SemanticModelSpec, SharedExpertSpec, StateSpaceSpec, StateSpaceStateDTypeSpec,
    },
};

pub(super) fn compile(
    decoder: &DecoderConfig,
    hybrid: &StateSpaceHybridConfig,
    norm: NormalizationSpec,
) -> Result<SemanticModelSpec> {
    let layers = hybrid
        .blocks
        .layers()
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            Ok(DecoderLayerSpec {
                index,
                input_norm: norm,
                post_attention_norm: norm,
                mixer: mixer(decoder, hybrid, *layer)?,
                feed_forward: match layer.feed_forward_block() {
                    Some(_) => latent_routed(hybrid),
                    None => FeedForwardSpec::None,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let prediction_layers = match hybrid.prediction {
        Some(prediction) => (0..prediction.depth.get())
            .map(|index| {
                Ok(DecoderLayerSpec {
                    index,
                    input_norm: norm,
                    post_attention_norm: norm,
                    mixer: attention(decoder)?,
                    feed_forward: latent_routed(hybrid),
                })
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    Ok(SemanticModelSpec {
        schema_version: CURRENT_SCHEMA_VERSION,
        decoder: DecoderSpec {
            hidden_size: decoder.hidden_size,
            vocab_size: decoder.vocab_size,
            tie_word_embeddings: decoder.tie_word_embeddings,
            final_norm: norm,
            layers,
            block_layout: BlockLayoutSpec::Paired {
                physical_blocks: hybrid.blocks.physical().len(),
                mixer_blocks: hybrid
                    .blocks
                    .layers()
                    .iter()
                    .map(|layer| layer.mixer_block())
                    .collect(),
            },
            prediction_layers,
        },
    })
}

fn mixer(
    decoder: &DecoderConfig,
    hybrid: &StateSpaceHybridConfig,
    layer: HybridLayer,
) -> Result<MixerSpec> {
    if layer.is_state_space() {
        let mamba = &hybrid.mamba;
        return Ok(MixerSpec::StateSpace(StateSpaceSpec {
            heads: mamba.heads,
            head_dim: mamba.head_dim,
            state_size: mamba.state_size,
            groups: mamba.groups,
            convolution_kernel_size: mamba.convolution_kernel_size,
            convolution_bias: mamba.convolution_bias,
            projection_bias: mamba.projection_bias,
            chunk_size: mamba.chunk_size,
            time_step_minimum: mamba.time_step_limit.minimum,
            time_step_maximum: match mamba.time_step_limit.maximum {
                TimeStepBound::Finite(value) => Some(value),
                TimeStepBound::Unbounded => None,
            },
            state_dtype: match mamba.state_dtype {
                StateSpaceStateDType::Float32 => StateSpaceStateDTypeSpec::Float32,
                StateSpaceStateDType::BFloat16 => StateSpaceStateDTypeSpec::BFloat16,
                StateSpaceStateDType::Float16 => StateSpaceStateDTypeSpec::Float16,
            },
        }));
    }
    attention(decoder)
}

/// The attention mixer of hybrid decoder and prediction layers.
fn attention(decoder: &DecoderConfig) -> Result<MixerSpec> {
    if decoder.sliding_window.is_some() {
        return Err(invalid("state-space hybrid attention does not support sliding windows"));
    }
    let head_dim = decoder.head_dim;
    Ok(MixerSpec::SoftmaxAttention(AttentionSpec {
        query_heads: decoder.num_attention_heads,
        key_value_heads: decoder.num_key_value_heads,
        head_dim,
        key_value_relation: KeyValueRelation::Separate,
        qk_normalization: QkNormalization::None,
        projection_bias: decoder.attention_bias,
        output: AttentionOutputSpec::Direct,
        sinks: false,
        scale: decoder.attention_scale.unwrap_or_else(|| {
            f64::from(u32::try_from(head_dim).unwrap_or(u32::MAX)).sqrt().recip()
        }),
        window: None,
        // The family relies on the recurrent mixers for order information.
        position: PositionEncodingSpec::None,
    }))
}

fn latent_routed(hybrid: &StateSpaceHybridConfig) -> FeedForwardSpec {
    let routed = &hybrid.routed;
    FeedForwardSpec::LatentRouted {
        latent_size: routed.latent_size,
        routed: RoutedExpertsSpec {
            expert_count: routed.expert_count,
            top_k: routed.top_k,
            intermediate_size: routed.expert_intermediate_size,
            activation: ActivationSpec::ReluSquared,
            router_normalization: RouterNormalization::SigmoidBiasedTopK,
        },
        shared: routed.shared_intermediate_size.map(|intermediate_size| SharedExpertSpec {
            intermediate_size,
            activation: ActivationSpec::ReluSquared,
            gated_output: false,
        }),
        routed_scale: routed.routed_scale,
    }
}
