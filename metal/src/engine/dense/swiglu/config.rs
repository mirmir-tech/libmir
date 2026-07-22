use models::layout::{DecoderConfig, RopeScaling};

use crate::engine::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub(super) struct DenseSwiGluLayerConfig {
    pub(super) heads: i32,
    pub(super) kv_heads: i32,
    pub(super) head_dim: i32,
    pub(super) attention_scale: f32,
    pub(super) rope_base: f32,
    pub(super) rope_scaling: Option<RopeScaling>,
    pub(super) rope_attention_factor: f32,
    pub(super) rms_norm_eps: f32,
    pub(super) group_size: i32,
}

impl DenseSwiGluLayerConfig {
    pub(super) fn from_decoder(decoder: &DecoderConfig, group_size: usize) -> Result<Self> {
        if decoder.num_experts.is_some() {
            return Err(Error::InvalidModel("dense SwiGLU path does not support MoE".into()));
        }
        if decoder.sliding_window.is_some() {
            return Err(Error::InvalidModel(
                "dense SwiGLU path does not support sliding-window attention".into(),
            ));
        }
        if decoder.hidden_activation.as_deref().is_some_and(|value| value != "silu") {
            return Err(Error::InvalidModel(format!(
                "dense SwiGLU path requires silu activation, found {:?}",
                decoder.hidden_activation
            )));
        }
        let head_dim = i32::try_from(decoder.head_dim)?;
        let attention_scale = head_dim.to_string().parse::<f32>()?.sqrt().recip();
        let config = Self {
            heads: i32::try_from(decoder.num_attention_heads)?,
            kv_heads: i32::try_from(decoder.num_key_value_heads)?,
            head_dim,
            attention_scale,
            rope_base: decoder.rope_theta.unwrap_or(10_000.0).to_string().parse()?,
            rope_scaling: decoder.rope_scaling,
            rope_attention_factor: decoder
                .rope_scaling
                .and_then(RopeScaling::yarn)
                .map_or(Ok(1.0), |values| values.4.to_string().parse())?,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            group_size: i32::try_from(group_size)?,
        };
        config.validate(decoder.hidden_size)?;
        Ok(config)
    }

    fn validate(self, hidden_size: usize) -> Result<()> {
        if [self.heads, self.kv_heads, self.head_dim, self.group_size]
            .into_iter()
            .any(|dimension| dimension <= 0)
        {
            return Err(Error::InvalidModel(format!(
                "non-positive dense SwiGLU dimensions: {self:?}"
            )));
        }
        let attention_width = i64::from(self.heads) * i64::from(self.head_dim);
        if attention_width != i64::try_from(hidden_size)? {
            return Err(Error::InvalidModel(format!(
                "dense SwiGLU attention width {attention_width} does not match hidden size {hidden_size}"
            )));
        }
        if !self.rope_base.is_finite()
            || !self.rms_norm_eps.is_finite()
            || !self.attention_scale.is_finite()
            || !self.rope_attention_factor.is_finite()
            || self.rope_attention_factor <= 0.0
        {
            return Err(Error::InvalidModel(format!("non-finite dense SwiGLU config: {self:?}")));
        }
        Ok(())
    }
}
