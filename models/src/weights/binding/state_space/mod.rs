//! Typed binding view of state-space hybrid decoder layers.

use super::{
    AttentionProjectionRole, LayerTensorRole, StateSpaceTensorRole, TensorBinding,
    WeightBindingPlan,
    slot::LayerSlot,
    view::{RoutedExpertBindings, sources},
};
use crate::error::{ModelsError, Result};

mod feed_forward;

/// Bindings of one state-space hybrid layer. A layer without a feed-forward
/// block has no feed-forward bindings, including no post-mixer norm.
#[derive(Debug, Clone, Copy)]
pub struct StateSpaceHybridLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub mixer: StateSpaceHybridMixerBindings<'a>,
    pub feed_forward: Option<StateSpaceHybridFeedForwardBindings<'a>>,
}

/// A hybrid layer's feed-forward block: routed experts, or a dense ungated
/// MLP (dense Nemotron-H).
#[derive(Debug, Clone, Copy)]
pub enum StateSpaceHybridFeedForwardBindings<'a> {
    Routed(LatentRoutedBindings<'a>),
    Dense(DenseHybridBindings<'a>),
}

impl<'a> StateSpaceHybridFeedForwardBindings<'a> {
    /// The routed-expert block, when the layer has one.
    #[must_use]
    pub const fn routed(self) -> Option<LatentRoutedBindings<'a>> {
        match self {
            Self::Routed(routed) => Some(routed),
            Self::Dense(_) => None,
        }
    }
}

/// A normalized ungated MLP: `down(act(up(norm(x))))`.
#[derive(Debug, Clone, Copy)]
pub struct DenseHybridBindings<'a> {
    pub norm: &'a TensorBinding,
    pub up: &'a TensorBinding,
    pub down: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub enum StateSpaceHybridMixerBindings<'a> {
    StateSpace(StateSpaceMixerBindings<'a>),
    Attention(AttentionProjectionBindings<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct StateSpaceMixerBindings<'a> {
    pub decay_log: &'a TensorBinding,
    pub skip_scale: &'a TensorBinding,
    /// Depthwise convolution; its bias, when present, is a dense companion.
    pub convolution: &'a TensorBinding,
    pub time_bias: &'a TensorBinding,
    pub input_projection: &'a TensorBinding,
    pub gated_norm: &'a TensorBinding,
    pub output_projection: &'a TensorBinding,
}

/// Separate Q/K/V/O projections without Q/K normalization.
#[derive(Debug, Clone, Copy)]
pub struct AttentionProjectionBindings<'a> {
    pub query: &'a TensorBinding,
    pub key: &'a TensorBinding,
    pub value: &'a TensorBinding,
    pub output: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct LatentRoutedBindings<'a> {
    pub norm: &'a TensorBinding,
    pub router: &'a TensorBinding,
    pub correction_bias: &'a TensorBinding,
    /// Projections into and out of the latent space; absent when routed
    /// experts operate at the model hidden width.
    pub latent: Option<LatentProjectionBindings<'a>>,
    /// Up/down projections, per expert or stacked, at the expert width.
    pub routed: RoutedExpertBindings<'a>,
    pub shared: Option<UngatedExpertBindings<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct LatentProjectionBindings<'a> {
    pub input: &'a TensorBinding,
    pub output: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct UngatedExpertBindings<'a> {
    pub up: &'a TensorBinding,
    pub down: &'a TensorBinding,
}

impl WeightBindingPlan {
    pub fn state_space_hybrid_layer(
        &self,
        index: usize,
    ) -> Result<StateSpaceHybridLayerBindings<'_>> {
        self.hybrid_layer(LayerSlot::Decoder(index))
    }

    /// The hybrid-layer bindings of a decoder layer or prediction module.
    pub(super) fn hybrid_layer(
        &self,
        slot: LayerSlot,
    ) -> Result<StateSpaceHybridLayerBindings<'_>> {
        let get = |tensor| required(self, slot, tensor);
        let mixer = if optional(self, slot, state(StateSpaceTensorRole::InputProjection)).is_some()
        {
            let get = |tensor| required(self, slot, state(tensor));
            StateSpaceHybridMixerBindings::StateSpace(StateSpaceMixerBindings {
                decay_log: get(StateSpaceTensorRole::DecayLog)?,
                skip_scale: get(StateSpaceTensorRole::SkipScale)?,
                convolution: get(StateSpaceTensorRole::Convolution)?,
                time_bias: get(StateSpaceTensorRole::TimeBias)?,
                input_projection: get(StateSpaceTensorRole::InputProjection)?,
                gated_norm: get(StateSpaceTensorRole::GatedNorm)?,
                output_projection: get(StateSpaceTensorRole::OutputProjection)?,
            })
        } else {
            let get = |projection| {
                required(self, slot, LayerTensorRole::AttentionProjection { projection })
            };
            StateSpaceHybridMixerBindings::Attention(AttentionProjectionBindings {
                query: get(AttentionProjectionRole::Query)?,
                key: get(AttentionProjectionRole::Key)?,
                value: get(AttentionProjectionRole::Value)?,
                output: get(AttentionProjectionRole::Output)?,
            })
        };
        let feed_forward = feed_forward::bindings(self, slot)?;
        Ok(StateSpaceHybridLayerBindings {
            input_norm: get(LayerTensorRole::InputNorm)?,
            mixer,
            feed_forward,
        })
    }
}

impl<'a> StateSpaceHybridLayerBindings<'a> {
    /// Every checkpoint tensor and companion this layer reads.
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = vec![self.input_norm];
        match self.mixer {
            StateSpaceHybridMixerBindings::StateSpace(mixer) => bindings.extend([
                mixer.decay_log,
                mixer.skip_scale,
                mixer.convolution,
                mixer.time_bias,
                mixer.input_projection,
                mixer.gated_norm,
                mixer.output_projection,
            ]),
            StateSpaceHybridMixerBindings::Attention(attention) => {
                bindings
                    .extend([attention.query, attention.key, attention.value, attention.output]);
            },
        }
        bindings.extend(self.feed_forward.into_iter().flat_map(feed_forward::sources));
        sources(bindings)
    }
}

const fn state(tensor: StateSpaceTensorRole) -> LayerTensorRole {
    LayerTensorRole::StateSpace { tensor }
}

pub(super) fn required(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    let role = slot.role(tensor);
    plan.binding(&role)
        .ok_or_else(|| invalid(slot, &format!("unbound role {role:?}")))
}

pub(super) fn optional(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&slot.role(tensor))
}

pub(super) fn invalid(slot: LayerSlot, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid state-space hybrid {slot}: {reason}"))
}
