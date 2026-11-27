//! The one layer a runtime built around a repeated layer reads instead of
//! the spec: attention with routed experts, the same in every layer.

use models::semantic::{
    DecoderLayerSpec, FeedForwardSpec, MixerSpec, PositionEncodingSpec, RopeScalingSpec,
    SemanticModelSpec,
};

use super::CudaLayer;
use crate::lowering::{Experts, FeedForwardLowering};

/// Geometry, norms and positions every layer shares, lowered with the plan
/// when the decoder repeats one softmax-attention layer with routed experts
/// rotated by full `YaRN`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepeatedLayer {
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
    /// Of every norm, the final one included.
    pub epsilon: f64,
    /// Each routed expert's.
    pub intermediate: usize,
    pub experts: Experts,
    pub rotary: YarnRotary,
}

/// Full standard `YaRN` rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnRotary {
    pub theta: f64,
    pub factor: f64,
    pub original_context: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub truncate: bool,
}

impl RepeatedLayer {
    /// The repeated layer of `spec` lowered as `layers`, if it repeats one.
    #[allow(clippy::float_cmp, reason = "repeated layers declare identical constants")]
    pub(super) fn of(spec: &SemanticModelSpec, layers: &[CudaLayer]) -> Option<Self> {
        let decoder = &spec.decoder;
        let mut repeated = None;
        for (layer, lowered) in decoder.layers.iter().zip(layers) {
            let current = Self::layer(layer, lowered)?;
            if repeated.is_some_and(|first| first != current) {
                return None;
            }
            repeated = Some(current);
        }
        repeated.filter(|layer| layer.epsilon == decoder.final_norm.epsilon)
    }

    #[allow(clippy::float_cmp, reason = "repeated layers declare identical constants")]
    fn layer(layer: &DecoderLayerSpec, lowered: &CudaLayer) -> Option<Self> {
        let (MixerSpec::SoftmaxAttention(attention), FeedForwardSpec::Routed { routed, .. }) =
            (&layer.mixer, &layer.feed_forward)
        else {
            return None;
        };
        let FeedForwardLowering::Routed { experts, shared: None } = lowered.feed_forward else {
            return None;
        };
        if layer.input_norm.epsilon != layer.post_attention_norm.epsilon {
            return None;
        }
        Some(Self {
            query_heads: attention.query_heads,
            key_value_heads: attention.key_value_heads,
            head_dim: attention.head_dim,
            scale: attention.scale,
            epsilon: layer.input_norm.epsilon,
            intermediate: routed.intermediate_size,
            experts,
            rotary: YarnRotary::of(&attention.position)?,
        })
    }
}

impl YarnRotary {
    /// `position` as full standard `YaRN`, if it is.
    #[allow(clippy::float_cmp, reason = "a full rotation declares exactly one")]
    #[must_use]
    pub fn of(position: &PositionEncodingSpec) -> Option<Self> {
        let PositionEncodingSpec::Rotary(rotary) = position else {
            return None;
        };
        let Some(RopeScalingSpec::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context_len,
            truncate,
            ..
        }) = rotary.scaling
        else {
            return None;
        };
        (rotary.layout == models::semantic::RotaryLayoutSpec::Standard
            && rotary.partial_factor == 1.0)
            .then_some(Self {
                theta: rotary.theta,
                factor,
                original_context: original_context_len,
                beta_fast,
                beta_slow,
                truncate,
            })
    }
}
