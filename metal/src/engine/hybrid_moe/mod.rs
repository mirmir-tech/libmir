mod attention;
mod batch;
mod feed_forward;
mod layer;
mod model;
mod weights;

pub use layer::HybridMoeLayer;
pub use model::HybridMoeModel;
use models::layout::{AttentionLayerType, DecoderConfig};

use super::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct HybridMoeLayerConfig {
    pub layer_index: usize,
    pub hidden_size: i32,
    pub attention_heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub rope_dimensions: i32,
    pub rope_base: f32,
    pub proportional_rope: bool,
    pub use_k_eq_v: bool,
    pub rms_norm_eps: f32,
    pub top_k: i32,
    pub group_size: i32,
    pub router_norm_scale: f32,
    pub max_context: Option<usize>,
}

impl HybridMoeLayerConfig {
    pub fn from_decoder(
        layer_index: usize,
        decoder: &DecoderConfig,
        group_size: usize,
    ) -> Result<Self> {
        let layer_type = decoder.layer_type(layer_index);
        let head_dim = decoder.layer_head_dim(layer_index);
        let partial = decoder.partial_rotary_factor_for_layer(layer_index).unwrap_or(1.0);
        let head_dim_f64 = head_dim.to_string().parse::<f64>()?;
        let rope_dimensions = float_to_i32((head_dim_f64 * partial).round())?;
        let hidden_size = i32::try_from(decoder.hidden_size)?;
        let router_norm_scale = 1.0 / hidden_size.to_string().parse::<f32>()?.sqrt();
        Ok(Self {
            layer_index,
            hidden_size,
            attention_heads: i32::try_from(decoder.num_attention_heads)?,
            kv_heads: i32::try_from(decoder.layer_key_value_heads(layer_index))?,
            head_dim: i32::try_from(head_dim)?,
            rope_dimensions,
            rope_base: decoder
                .rope_theta_for_layer(layer_index)
                .unwrap_or(10_000.0)
                .to_string()
                .parse()?,
            proportional_rope: decoder.rope_type_for_layer(layer_index) == Some("proportional"),
            use_k_eq_v: layer_type == AttentionLayerType::Full && decoder.attention_k_eq_v,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            top_k: i32::try_from(decoder.top_k_experts.unwrap_or(1))?,
            group_size: i32::try_from(group_size)?,
            router_norm_scale,
            max_context: decoder.layer_sliding_window(layer_index),
        })
    }

    pub(super) fn validate(self) -> Result<Self> {
        let dimensions = [
            self.hidden_size,
            self.attention_heads,
            self.kv_heads,
            self.head_dim,
            self.rope_dimensions,
            self.top_k,
            self.group_size,
        ];
        if dimensions.into_iter().any(|dimension| dimension <= 0) {
            return Err(Error::InvalidModel(format!("non-positive dimensions: {self:?}")));
        }
        if !self.rope_base.is_finite()
            || !self.rms_norm_eps.is_finite()
            || !self.router_norm_scale.is_finite()
        {
            return Err(Error::InvalidModel(format!("non-finite parameters: {self:?}")));
        }
        Ok(self)
    }
}

fn float_to_i32(value: f64) -> Result<i32> {
    Ok(value.to_string().parse::<i32>()?)
}
