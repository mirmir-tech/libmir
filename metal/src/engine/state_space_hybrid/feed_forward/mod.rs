//! The feed-forward block of a hybrid layer: sigmoid-routed `relu²` experts,
//! or a dense ungated `relu²` MLP (dense Nemotron-H).

mod routed;

use models::{semantic::DecoderLayerSpec, weights::StateSpaceHybridFeedForwardBindings};
pub(super) use routed::RoutedFeedForward;

use crate::engine::{Array, Error, ModelTensors, NormWeight, Result, Stream, binding::BoundLinear};

#[derive(Debug)]
pub(super) enum FeedForward {
    Routed(Box<RoutedFeedForward>),
    Dense(Box<DenseFeedForward>),
}

/// `down(relu²(up(x)))` behind its block norm.
#[derive(Debug)]
pub(super) struct DenseFeedForward {
    norm: NormWeight,
    up: BoundLinear,
    down: BoundLinear,
    epsilon: f32,
}

impl FeedForward {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: StateSpaceHybridFeedForwardBindings<'_>,
        layer: &DecoderLayerSpec,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(match bindings {
            StateSpaceHybridFeedForwardBindings::Routed(routed) => {
                let spec = layer.feed_forward.routed().ok_or_else(|| {
                    Error::InvalidModel(format!(
                        "hybrid block {} has no routed experts",
                        layer.index
                    ))
                })?;
                Self::Routed(Box::new(RoutedFeedForward::load(
                    tensors, routed, spec, epsilon, stream,
                )?))
            },
            StateSpaceHybridFeedForwardBindings::Dense(dense) => {
                Self::Dense(Box::new(DenseFeedForward {
                    norm: NormWeight::load_name(tensors, &dense.norm.source)?,
                    up: BoundLinear::load(tensors, dense.up, stream)?,
                    down: BoundLinear::load(tensors, dense.down, stream)?,
                    epsilon,
                }))
            },
        })
    }

    /// The block norm and its epsilon.
    pub(super) fn norm(&self) -> (&NormWeight, f32) {
        match self {
            Self::Routed(routed) => (&routed.norm, routed.epsilon()),
            Self::Dense(dense) => (&dense.norm, dense.epsilon),
        }
    }

    /// Runs the block over a normalized `[rows, sequence, hidden]` input.
    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Routed(routed) => routed.forward(input, stream),
            Self::Dense(dense) => dense
                .down
                .forward(&dense.up.forward(input, stream)?.relu_squared(stream)?, stream),
        }
    }
}
