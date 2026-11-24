use models::semantic::{
    ActivationSpec, AttentionSpec, DecoderLayerSpec, DecoderSpec, PositionEncodingSpec,
};

use crate::engine::{Error, Result};

/// GPT-OSS's `x · σ(1.702 x)` gate, which the kernels fix.
const GATE_ALPHA: f64 = 1.702;

#[derive(Debug, Clone, Copy)]
pub(super) struct ClampedRoutedConfig {
    pub hidden: i32,
    pub intermediate: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub top_k: i32,
    pub epsilon: f32,
    pub scale: f32,
    /// `YaRN`'s factor on the rotated cosine and sine.
    pub rope_concentration: f32,
    pub swiglu_limit: f32,
}

impl ClampedRoutedConfig {
    pub fn from_spec(decoder: &DecoderSpec, layer: &DecoderLayerSpec) -> Result<Self> {
        let attention = attention(layer)?;
        let routed = layer.feed_forward.routed().ok_or_else(|| {
            Error::InvalidModel("clamped-routed layers need routed experts".into())
        })?;
        let ActivationSpec::ClampedSwiGlu { alpha, limit } = routed.activation else {
            return Err(Error::InvalidModel("clamped-routed experts need a clamped SwiGLU".into()));
        };
        if alpha.to_bits() != GATE_ALPHA.to_bits() {
            return Err(Error::InvalidModel("unsupported clamped SwiGLU constants".into()));
        }
        let config = Self {
            hidden: i32::try_from(decoder.hidden_size)?,
            intermediate: i32::try_from(routed.intermediate_size)?,
            heads: i32::try_from(attention.query_heads)?,
            kv_heads: i32::try_from(attention.key_value_heads)?,
            head_dim: i32::try_from(attention.head_dim)?,
            top_k: i32::try_from(routed.top_k)?,
            epsilon: decimal(layer.input_norm.epsilon)?,
            scale: decimal(attention.scale)?,
            rope_concentration: decimal(concentration(attention)?)?,
            swiglu_limit: decimal(limit)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<()> {
        let dimensions =
            [self.hidden, self.intermediate, self.heads, self.kv_heads, self.head_dim, self.top_k];
        if dimensions.into_iter().any(|value| value <= 0)
            || self.hidden % 32 != 0
            || self.intermediate % 32 != 0
            || self.heads % self.kv_heads != 0
            || !self.scale.is_finite()
            || !self.swiglu_limit.is_finite()
        {
            Err(Error::InvalidModel("invalid clamped-routed configuration".into()))
        } else {
            Ok(())
        }
    }
}

pub(super) fn attention(layer: &DecoderLayerSpec) -> Result<&AttentionSpec> {
    layer.mixer.softmax_attention().ok_or_else(|| {
        Error::InvalidModel(format!("clamped-routed layer {} needs softmax attention", layer.index))
    })
}

/// The rotation's `YaRN` attention factor; the encoding must rotate whole
/// heads with scaled frequencies.
fn concentration(attention: &AttentionSpec) -> Result<f64> {
    let PositionEncodingSpec::Rotary(rotary) = &attention.position else {
        return Err(Error::InvalidModel("clamped-routed attention needs RoPE".into()));
    };
    if rotary.scaling.is_none()
        || rotary.rotated_dimensions(attention.head_dim)? != attention.head_dim
    {
        return Err(Error::InvalidModel(
            "clamped-routed execution requires full-head scaled RoPE".into(),
        ));
    }
    let (_, factor) = attention
        .position
        .inverse_frequencies(attention.head_dim)?
        .ok_or_else(|| Error::InvalidModel("clamped-routed attention needs RoPE".into()))?;
    Ok(factor)
}

fn decimal(value: f64) -> Result<f32> {
    Ok(value.to_string().parse()?)
}
