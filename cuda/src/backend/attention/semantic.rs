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
            stream_norm: crate::kernels::NormFormula::Rms,
            attention_scale: narrow("attention scale", attention.scale)?,
            projection_format: policy.projection_format,
            qkv_normalization: attention.qk_normalization.into(),
            window: attention.window,
            score_softcap: attention
                .logit_softcap
                .map(|cap| crate::kernels::ScoreSoftcap::new(narrow("attention score cap", cap)?))
                .transpose()?,
            query_temperature: attention
                .query_temperature
                .map(crate::kernels::QueryTemperatureScale::new)
                .transpose()?,
            max_sequence_blocks: policy.max_sequence_blocks,
            cache: KvStorageSpec::new(policy.cache, attention.key_value_heads, attention.head_dim),
        })
    }
}

impl From<NormalizationKind> for crate::kernels::NormFormula {
    fn from(kind: NormalizationKind) -> Self {
        match kind {
            NormalizationKind::Rms => Self::Rms,
            NormalizationKind::Layer => Self::MeanCentred,
        }
    }
}

impl From<QkNormalization> for QkvNormalization {
    fn from(normalization: QkNormalization) -> Self {
        match normalization {
            QkNormalization::None => Self::NONE,
            QkNormalization::QueryKeyRms => Self::QUERY_KEY,
            // RoPE rotates pairs within a head, so a unit RMS norm before it
            // equals Llama 4's norm after it.
            QkNormalization::QueryKeyUnitRms => Self {
                weights: crate::kernels::QkNormWeights::Unit,
                ..Self::QUERY_KEY
            },
            QkNormalization::QueryKeyRmsAcrossHeads => Self::QUERY_KEY_ACROSS_HEADS,
            QkNormalization::QueryKeyValueRms => Self::ALL,
        }
    }
}

fn rotary(position: &PositionEncodingSpec, head_dim: usize) -> Result<Option<RotaryEmbedding>> {
    let rotary = match position {
        PositionEncodingSpec::None => return Ok(None),
        PositionEncodingSpec::Rotary(rotary) => rotary,
    };
    let RotarySpec {
        theta,
        layout: RotaryLayoutSpec::Standard,
        algorithm: None,
        scaling: None,
        ..
    } = rotary
    else {
        return Err(unsupported("only standard unscaled rotary embeddings are implemented"));
    };
    let dimensions = rotary.rotated_dimensions(head_dim)?;
    Ok(Some(RotaryEmbedding {
        dimensions,
        pairing: dimensions,
        theta: narrow("RoPE theta", *theta)?,
        scaling: crate::kernels::RotaryScaling::NONE,
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
