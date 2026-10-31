//! Decode attention configuration derived from a semantic decoder layer.

use models::semantic::{
    AttentionOutputSpec, DecoderLayerSpec, KeyValueRelation, MixerSpec, NormalizationKind,
    PositionEncodingSpec, QkNormalization, RotaryLayoutSpec, RotarySpec,
};
use runtime::kv::{CacheConfig, KvStorageSpec};

use super::DecodeAttentionConfig;
use crate::{
    Error, ProjectionFormat, Result,
    kernels::{QkvNormalization, RotaryEmbedding},
};

/// Model-wide placement and storage policy of softmax-attention layers.
#[derive(Clone, Copy, Debug)]
pub struct AttentionLayerPolicy {
    pub hidden_size: usize,
    pub projection_format: ProjectionFormat,
    pub cache: CacheConfig,
    pub max_sequence_blocks: usize,
}

impl DecodeAttentionConfig {
    /// Derives the decode attention of a semantic softmax-attention layer
    /// whose K/V state lives in cache layer `layer`.
    pub fn from_semantic(
        layer: usize,
        spec: &DecoderLayerSpec,
        policy: AttentionLayerPolicy,
    ) -> Result<Self> {
        let MixerSpec::SoftmaxAttention(attention) = &spec.mixer else {
            return Err(unsupported("layer mixer is not softmax attention"));
        };
        if spec.input_norm.kind != NormalizationKind::Rms
            || attention.key_value_relation != KeyValueRelation::Separate
            || attention.output != AttentionOutputSpec::Direct
            || attention.projection_bias
            || attention.sinks
        {
            return Err(unsupported("attention variant is not implemented by decode attention"));
        }
        Ok(Self {
            layer,
            hidden_size: policy.hidden_size,
            query_heads: attention.query_heads,
            rotary: rotary(&attention.position, attention.head_dim)?,
            rms_norm_epsilon: narrow("RMS norm epsilon", spec.input_norm.epsilon)?,
            attention_scale: narrow("attention scale", attention.scale)?,
            projection_format: policy.projection_format,
            qkv_normalization: match attention.qk_normalization {
                QkNormalization::None => QkvNormalization::NONE,
                QkNormalization::QueryKeyRms => QkvNormalization::QUERY_KEY,
            },
            sliding_window: attention.window,
            max_sequence_blocks: policy.max_sequence_blocks,
            cache: KvStorageSpec::new(policy.cache, attention.key_value_heads, attention.head_dim),
        })
    }
}

fn rotary(position: &PositionEncodingSpec, head_dim: usize) -> Result<Option<RotaryEmbedding>> {
    let rotary = match position {
        PositionEncodingSpec::None => return Ok(None),
        PositionEncodingSpec::Rotary(rotary) => rotary,
    };
    let RotarySpec {
        theta,
        partial_factor,
        layout: RotaryLayoutSpec::Standard,
        algorithm: None,
        scaling: None,
    } = rotary
    else {
        return Err(unsupported("only standard unscaled rotary embeddings are implemented"));
    };
    let dimensions = f64::from(u32::try_from(head_dim)?) * partial_factor;
    if !(dimensions > 0.0 && dimensions.fract() == 0.0) {
        return Err(unsupported("partial rotary factor does not produce an integral dimension"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let dimensions = dimensions as usize;
    Ok(Some(RotaryEmbedding {
        dimensions,
        pairing: dimensions,
        theta: narrow("RoPE theta", *theta)?,
    }))
}

fn narrow(label: &str, value: f64) -> Result<f32> {
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(unsupported(&format!("{label} is outside finite f32 range")));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(value as f32)
}

fn unsupported(message: &str) -> Error {
    Error::UnsupportedDecoderLayer(message.into())
}
