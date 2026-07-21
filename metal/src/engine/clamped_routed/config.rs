use models::layout::DecoderConfig;

use crate::engine::{Error, Result};

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
    pub rope_base: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_context: i32,
    pub rope_concentration: f32,
    pub swiglu_limit: f32,
}

impl ClampedRoutedConfig {
    pub fn from_decoder(decoder: &DecoderConfig) -> Result<Self> {
        let (factor, beta_fast, beta_slow, original) = decoder
            .rope_scaling
            .and_then(models::layout::RopeScaling::yarn)
            .ok_or_else(|| {
                Error::InvalidModel("clamped-routed execution requires YaRN configuration".into())
            })?;
        let head_dim = decoder.head_dim.to_string().parse::<f32>()?;
        let factor_f32 = factor.to_string().parse::<f32>()?;
        let config = Self {
            hidden: i32::try_from(decoder.hidden_size)?,
            intermediate: i32::try_from(decoder.intermediate_size)?,
            heads: i32::try_from(decoder.num_attention_heads)?,
            kv_heads: i32::try_from(decoder.num_key_value_heads)?,
            head_dim: i32::try_from(decoder.head_dim)?,
            top_k: i32::try_from(decoder.top_k_experts.unwrap_or_default())?,
            epsilon: decoder.rms_norm_eps.to_string().parse()?,
            scale: match decoder.attention_scale {
                Some(value) => value.to_string().parse()?,
                None => 1.0 / head_dim.sqrt(),
            },
            rope_base: decoder.rope_theta.unwrap_or(150_000.0).to_string().parse()?,
            rope_factor: factor_f32,
            beta_fast: beta_fast.to_string().parse()?,
            beta_slow: beta_slow.to_string().parse()?,
            original_context: i32::try_from(original)?,
            rope_concentration: 0.1_f32.mul_add(factor_f32.ln(), 1.0),
            swiglu_limit: decoder.swiglu_limit.unwrap_or(7.0).to_string().parse()?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<()> {
        let dimensions = [
            self.hidden,
            self.intermediate,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.top_k,
            self.original_context,
        ];
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
