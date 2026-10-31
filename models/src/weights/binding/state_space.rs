//! Typed binding view of state-space hybrid decoder layers.

use super::{
    AttentionProjectionRole, FeedForwardProjectionRole, LatentProjectionRole, LayerTensorRole,
    LogicalTensorRole, StateSpaceTensorRole, TensorBinding, WeightBindingPlan,
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
    pub latent_input: &'a TensorBinding,
    pub latent_output: &'a TensorBinding,
    /// Per-expert up/down projections at latent width.
    pub routed: RoutedExpertBindings<'a>,
    pub shared: Option<UngatedExpertBindings<'a>>,
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
        let get = |tensor| required(self, index, tensor);
        let mixer = if optional(self, index, state(StateSpaceTensorRole::InputProjection)).is_some()
        {
            let get = |tensor| required(self, index, state(tensor));
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
                required(self, index, LayerTensorRole::AttentionProjection { projection })
            };
            StateSpaceHybridMixerBindings::Attention(AttentionProjectionBindings {
                query: get(AttentionProjectionRole::Query)?,
                key: get(AttentionProjectionRole::Key)?,
                value: get(AttentionProjectionRole::Value)?,
                output: get(AttentionProjectionRole::Output)?,
            })
        };
        let feed_forward = optional(self, index, LayerTensorRole::Router)
            .map(|router| latent_routed(self, index, router))
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
            bindings.extend([
                feed_forward.norm,
                feed_forward.router,
                feed_forward.correction_bias,
                feed_forward.latent_input,
                feed_forward.latent_output,
            ]);
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
    index: usize,
    router: &'a TensorBinding,
) -> Result<LatentRoutedBindings<'a>> {
    let get = |tensor| required(plan, index, tensor);
    let latent = |projection| get(LayerTensorRole::LatentProjection { projection });
    let shared =
        |projection| optional(plan, index, LayerTensorRole::SharedExpertProjection { projection });
    let shared =
        match (shared(FeedForwardProjectionRole::Up), shared(FeedForwardProjectionRole::Down)) {
            (Some(up), Some(down)) => Some(UngatedExpertBindings { up, down }),
            (None, None) => None,
            _ => return Err(invalid(index, "shared expert has only one projection")),
        };
    Ok(LatentRoutedBindings {
        norm: get(LayerTensorRole::PostAttentionNorm)?,
        router,
        correction_bias: get(LayerTensorRole::RouterCorrectionBias)?,
        latent_input: latent(LatentProjectionRole::Input)?,
        latent_output: latent(LatentProjectionRole::Output)?,
        routed: RoutedExpertBindings::Individual { tensors: &plan.tensors, layer: index },
        shared,
    })
}

const fn state(tensor: StateSpaceTensorRole) -> LayerTensorRole {
    LayerTensorRole::StateSpace { tensor }
}

fn required(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    let role = LogicalTensorRole::Layer { index, tensor };
    plan.binding(&role)
        .ok_or_else(|| invalid(index, &format!("unbound role {role:?}")))
}

fn optional(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&LogicalTensorRole::Layer { index, tensor })
}

fn invalid(index: usize, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid state-space hybrid layer {index}: {reason}"))
}
