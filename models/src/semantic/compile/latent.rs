//! Semantic compilation of latent-attention decoders.

use super::{CURRENT_SCHEMA_VERSION, invalid, rotary};
use crate::{
    error::Result,
    layout::{AttentionLayerType, DecoderConfig, LatentDecoderConfig, RotaryPairing},
    semantic::{
        BlockLayoutSpec, DecoderLayerSpec, DecoderSpec, LatentAttentionSpec, MixerSpec,
        NormalizationSpec, PositionEncodingSpec, RotaryPairingSpec, SemanticModelSpec,
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
                norm_placement: crate::semantic::NormPlacement::Pre,
                mixer: mixer(decoder, latent, index, norm)?,
                feed_forward: super::experts::feed_forward(decoder, index)?,
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
            residual_multiplier: decoder.residual_multiplier.unwrap_or(1.0),
            logit_divisor: decoder.logits_divisor.unwrap_or(1.0),
            layers,
            block_layout: BlockLayoutSpec::Unified,
            prediction_layers: Vec::new(),
            per_layer_input: None,
        },
    })
}

impl LatentAttentionSpec {
    /// The latent attention of layer `index` of a latent decoder, as the
    /// semantic model declares it.
    pub fn for_layer(decoder: &DecoderConfig, index: usize) -> Result<Self> {
        let latent = decoder.latent.as_ref().ok_or_else(|| invalid("not a latent decoder"))?;
        let norm = NormalizationSpec {
            kind: crate::semantic::NormalizationKind::Rms,
            epsilon: decoder.rms_norm_eps,
            weight_offset: 0.0,
        };
        match mixer(decoder, latent, index, norm)? {
            MixerSpec::LatentAttention(spec) => Ok(spec),
            _ => Err(invalid("latent decoder layer without latent attention")),
        }
    }
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
        query_temperature: attention.query_temperature,
    }))
}
