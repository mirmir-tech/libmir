mod attention;
mod batch;
#[cfg(test)]
mod checkpoint;
mod feed_forward;
mod layer;
mod model;
mod sharing;
mod weights;

use std::num::NonZeroUsize;

#[cfg(test)]
pub use checkpoint::CheckpointLayer;
pub use layer::HybridMoeLayer;
pub use model::HybridMoeModel;
use models::semantic::{
    AttentionSpec, AttentionWindow, DecoderLayerSpec, DecoderSpec, KeyValueRelation,
    PositionEncodingSpec,
};

use super::super::{Error, Result};

/// The routed experts' shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedShape {
    pub top_k: i32,
    pub count: usize,
    pub intermediate: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridMoeLayerConfig {
    pub layer_index: usize,
    pub hidden_size: i32,
    pub attention_heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    /// Dimensions the rotation pairs over (the whole head for proportional
    /// `RoPE`, whose unrotated pairs carry infinite periods).
    pub rope_dimensions: i32,
    pub rope_base: f32,
    /// The fused kernel's position scale: `1 / factor` under linear scaling.
    pub rope_position_scale: f32,
    /// Rotate adjacent pairs: llama.cpp stores Llama-family Q/K rows
    /// interleaved.
    pub rope_traditional: bool,
    pub use_k_eq_v: bool,
    pub rms_norm_eps: f32,
    /// `None` for dense GELU layers.
    pub experts: Option<RoutedShape>,
    pub group_size: i32,
    pub router_norm_scale: f32,
    /// The sliding window of a local layer.
    pub sliding_window: Option<NonZeroUsize>,
}

impl HybridMoeLayerConfig {
    pub fn from_spec(
        decoder: &DecoderSpec,
        layer: &DecoderLayerSpec,
        interleaved_rows: bool,
        group_size: usize,
    ) -> Result<Self> {
        let attention = attention(layer)?;
        let PositionEncodingSpec::Rotary(rotary) = &attention.position else {
            return Err(Error::InvalidModel("hybrid MoE attention needs RoPE".into()));
        };
        let hidden_size = i32::try_from(decoder.hidden_size)?;
        let experts = layer
            .feed_forward
            .routed()
            .map(|routed| {
                Ok::<_, Error>(RoutedShape {
                    top_k: i32::try_from(routed.top_k)?,
                    count: routed.expert_count,
                    intermediate: routed.intermediate_size,
                })
            })
            .transpose()?;
        Self {
            layer_index: layer.index,
            hidden_size,
            attention_heads: i32::try_from(attention.query_heads)?,
            kv_heads: i32::try_from(attention.key_value_heads)?,
            head_dim: i32::try_from(attention.head_dim)?,
            rope_dimensions: i32::try_from(rotary.pairing_dimensions(attention.head_dim)?)?,
            rope_base: rotary.theta.to_string().parse()?,
            rope_position_scale: crate::engine::attention::rope_position_scale(attention)?,
            rope_traditional: interleaved_rows,
            use_k_eq_v: attention.key_value_relation == KeyValueRelation::KeyEqualsValue,
            rms_norm_eps: layer.input_norm.epsilon.to_string().parse()?,
            experts,
            group_size: i32::try_from(group_size)?,
            router_norm_scale: 1.0 / hidden_size.to_string().parse::<f32>()?.sqrt(),
            sliding_window: match attention.window {
                Some(AttentionWindow::Sliding(window)) => Some(window),
                Some(AttentionWindow::Chunked { .. }) => {
                    return Err(Error::InvalidModel("hybrid MoE has no chunked attention".into()));
                },
                None => None,
            },
        }
        .validate()
    }

    /// The tokens a local layer's cache keeps; `None` keeps the history.
    pub fn max_context(self) -> Option<usize> {
        self.sliding_window.map(NonZeroUsize::get)
    }

    /// The routed experts' shape; dense GELU layers have none.
    pub(in crate::engine) fn experts(self) -> Result<RoutedShape> {
        self.experts.ok_or_else(|| {
            Error::InvalidModel(format!("hybrid MoE layer {} has no experts", self.layer_index))
        })
    }

    pub(in crate::engine) fn validate(self) -> Result<Self> {
        let dimensions = [
            self.hidden_size,
            self.attention_heads,
            self.kv_heads,
            self.head_dim,
            self.rope_dimensions,
            self.group_size,
        ];
        if dimensions.into_iter().any(|dimension| dimension <= 0)
            || self.experts.is_some_and(|experts| {
                experts.top_k <= 0 || experts.count == 0 || experts.intermediate == 0
            })
        {
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

pub(in crate::engine) fn attention(layer: &DecoderLayerSpec) -> Result<&AttentionSpec> {
    layer.mixer.softmax_attention().ok_or_else(|| {
        Error::InvalidModel(format!("hybrid MoE layer {} needs softmax attention", layer.index))
    })
}
