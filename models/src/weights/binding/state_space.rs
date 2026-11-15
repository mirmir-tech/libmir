//! Typed binding view of state-space hybrid decoder layers.

use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LatentProjectionRole,
    LayerTensorRole, StateSpaceTensorRole, TensorBinding, WeightBindingPlan,
    slot::LayerSlot,
    view::{RoutedExpertBindings, sources},
};
use crate::error::{ModelsError, Result};

/// Bindings of one state-space hybrid layer. A layer without a routed block
/// has no feed-forward bindings, including no post-mixer norm.
#[derive(Debug, Clone, Copy)]
pub struct StateSpaceHybridLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub mixer: StateSpaceHybridMixerBindings<'a>,
    pub feed_forward: Option<LatentRoutedBindings<'a>>,
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
        let feed_forward = optional(self, slot, LayerTensorRole::Router)
            .map(|router| latent_routed(self, slot, router))
            .transpose()?;
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
        if let Some(feed_forward) = self.feed_forward {
            bindings.extend([feed_forward.norm, feed_forward.router, feed_forward.correction_bias]);
            if let Some(latent) = feed_forward.latent {
                bindings.extend([latent.input, latent.output]);
            }
            bindings.extend(feed_forward.routed.bindings());
            if let Some(shared) = feed_forward.shared {
                bindings.extend([shared.up, shared.down]);
            }
        }
        sources(bindings)
    }
}

fn latent_routed<'a>(
    plan: &'a WeightBindingPlan,
    slot: LayerSlot,
    router: &'a TensorBinding,
) -> Result<LatentRoutedBindings<'a>> {
    let get = |tensor| required(plan, slot, tensor);
    let projection =
        |projection| optional(plan, slot, LayerTensorRole::LatentProjection { projection });
    let latent = match (
        projection(LatentProjectionRole::Input),
        projection(LatentProjectionRole::Output),
    ) {
        (Some(input), Some(output)) => Some(LatentProjectionBindings { input, output }),
        (None, None) => None,
        _ => return Err(invalid(slot, "latent space has only one projection")),
    };
    let stacked = |projection| {
        optional(plan, slot, LayerTensorRole::ExpertProjection { expert: None, projection })
    };
    let experts = match (stacked(ExpertProjectionRole::Up), stacked(ExpertProjectionRole::Down)) {
        (Some(up), Some(down)) => RoutedExpertBindings::UngatedStacked { up, down },
        (None, None) => RoutedExpertBindings::Individual { tensors: &plan.tensors, layer: slot },
        _ => return Err(invalid(slot, "stacked experts have only one projection")),
    };
    let shared =
        |projection| optional(plan, slot, LayerTensorRole::SharedExpertProjection { projection });
    let shared =
        match (shared(FeedForwardProjectionRole::Up), shared(FeedForwardProjectionRole::Down)) {
            (Some(up), Some(down)) => Some(UngatedExpertBindings { up, down }),
            (None, None) => None,
            _ => return Err(invalid(slot, "shared expert has only one projection")),
        };
    Ok(LatentRoutedBindings {
        norm: get(LayerTensorRole::PostAttentionNorm)?,
        router,
        correction_bias: get(LayerTensorRole::RouterCorrectionBias)?,
        latent,
        routed: experts,
        shared,
    })
}

const fn state(tensor: StateSpaceTensorRole) -> LayerTensorRole {
    LayerTensorRole::StateSpace { tensor }
}

fn required(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    let role = slot.role(tensor);
    plan.binding(&role)
        .ok_or_else(|| invalid(slot, &format!("unbound role {role:?}")))
}

fn optional(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&slot.role(tensor))
}

fn invalid(slot: LayerSlot, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid state-space hybrid {slot}: {reason}"))
}
