use models::{layout::DecoderConfig, weights::MixedFeedForwardBindings};

use super::super::feed_forward::{DenseMlp, FeedForward};
use crate::engine::{ModelTensors, Result, SharedExpertMoe, Stream, lowering::LayerLowering};

/// A dense `SwiGLU` or routed plus shared experts.
pub(super) fn load(
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    bindings: MixedFeedForwardBindings<'_>,
    lowering: LayerLowering,
    stream: &Stream,
) -> Result<FeedForward> {
    Ok(match bindings {
        MixedFeedForwardBindings::Dense(dense) => {
            FeedForward::Dense(Box::new(DenseMlp::load(tensors, dense, stream)?))
        },
        MixedFeedForwardBindings::SharedRouted(routed) => FeedForward::Routed(Box::new(
            SharedExpertMoe::load_decoder(tensors, decoder, routed, lowering.feed_forward, stream)?,
        )),
    })
}
