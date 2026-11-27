//! Gated full attention's shape and rotation, from the layer's semantic
//! specification.

use models::semantic::{
    AttentionOutputSpec, AttentionSpec, PositionEncodingSpec, RotaryLayoutSpec,
};

use crate::engine::{Error, Result};

#[derive(Debug, Clone)]
pub struct GatedFullAttentionConfig {
    pub(super) attention_heads: i32,
    pub(super) key_value_heads: i32,
    pub(super) head_dim: i32,
    pub(super) rope_dimensions: i32,
    pub(super) rope_base: f32,
    pub(super) attention_scale: f32,
    pub(super) rms_norm_eps: f32,
    pub(super) rope_layout: RotaryLayoutSpec,
}

impl GatedFullAttentionConfig {
    /// The layer's attention from its semantic specification; its query
    /// and key norms use `norm_epsilon`.
    pub fn from_spec(attention: &AttentionSpec, norm_epsilon: f64) -> Result<Self> {
        if attention.output != AttentionOutputSpec::Gated {
            return Err(Error::InvalidModel("gated full attention requires an output gate".into()));
        }
        let head_dim = attention.head_dim;
        let rotary = match &attention.position {
            PositionEncodingSpec::Rotary(rotary)
                if rotary.scaling.is_none()
                    && rotary.pairing_dimensions(head_dim)?
                        == rotary.rotated_dimensions(head_dim)? =>
            {
                rotary
            },
            _ => {
                return Err(Error::InvalidModel("gated full attention requires plain RoPE".into()));
            },
        };
        let config = Self {
            attention_heads: i32::try_from(attention.query_heads)?,
            key_value_heads: i32::try_from(attention.key_value_heads)?,
            head_dim: i32::try_from(attention.head_dim)?,
            rope_dimensions: i32::try_from(rotary.rotated_dimensions(attention.head_dim)?)?,
            rope_base: rotary.theta.to_string().parse()?,
            attention_scale: attention.scale.to_string().parse()?,
            rms_norm_eps: norm_epsilon.to_string().parse()?,
            rope_layout: rotary.layout.clone(),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.attention_heads <= 0
            || self.key_value_heads <= 0
            || self.head_dim <= 0
            || self.attention_heads % self.key_value_heads != 0
            || self.rope_dimensions <= 0
            || self.rope_dimensions > self.head_dim
            || self.rope_dimensions % 2 != 0
            || !self.rope_base.is_finite()
            || !self.attention_scale.is_finite()
            || !self.rms_norm_eps.is_finite()
        {
            return Err(Error::InvalidModel(format!(
                "invalid gated full attention configuration: {self:?}"
            )));
        }
        Ok(())
    }
}
