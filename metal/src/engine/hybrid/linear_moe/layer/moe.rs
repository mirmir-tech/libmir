use models::{semantic::DecoderLayerSpec, weights::MixedFeedForwardBindings};

use super::super::feed_forward::{DenseMlp, FeedForward};
use crate::engine::{
    Error, ModelTensors, Result, SharedExpertMoe, Stream, lowering::LayerLowering,
};

/// A dense `SwiGLU` or routed plus shared experts.
pub(super) fn load(
    tensors: &ModelTensors,
    layer: &DecoderLayerSpec,
    bindings: MixedFeedForwardBindings<'_>,
    lowering: LayerLowering,
    stream: &Stream,
) -> Result<FeedForward> {
    Ok(match bindings {
        MixedFeedForwardBindings::Dense(dense) => {
            FeedForward::Dense(Box::new(DenseMlp::load(tensors, dense, stream)?))
        },
        MixedFeedForwardBindings::SharedRouted(routed) => {
            FeedForward::Routed(Box::new(SharedExpertMoe::load_spec(
                tensors,
                layer.feed_forward.routed().ok_or_else(|| {
                    Error::InvalidModel("shared experts need routed experts".into())
                })?,
                routed,
                lowering.feed_forward,
                stream,
            )?))
        },
    })
}
