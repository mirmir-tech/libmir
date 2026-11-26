use models::{
    semantic::{DecoderLayerSpec, FeedForwardSpec},
    weights::MixedFeedForwardBindings,
};

use super::super::feed_forward::{DenseMlp, FeedForward};
use crate::engine::{
    Error, ModelTensors, Result, SharedExpertMoe, Stream, mlp_gate::GateActivation,
};

/// A dense gated MLP, gated as the layer's activation says, or routed plus
/// shared experts.
pub(super) fn load(
    tensors: &ModelTensors,
    layer: &DecoderLayerSpec,
    bindings: MixedFeedForwardBindings<'_>,
    stream: &Stream,
) -> Result<FeedForward> {
    Ok(match bindings {
        MixedFeedForwardBindings::Dense(dense) => {
            let FeedForwardSpec::Dense { activation, .. } = &layer.feed_forward else {
                return Err(Error::InvalidModel(format!(
                    "dense layer {} has no dense feed-forward",
                    layer.index
                )));
            };
            let activation = GateActivation::from_spec(activation)?;
            FeedForward::Dense(Box::new(DenseMlp::load(tensors, dense, activation, stream)?))
        },
        MixedFeedForwardBindings::SharedRouted(routed) => {
            FeedForward::Routed(Box::new(SharedExpertMoe::load_spec(
                tensors,
                layer.feed_forward.routed().ok_or_else(|| {
                    Error::InvalidModel("shared experts need routed experts".into())
                })?,
                routed,
                stream,
            )?))
        },
    })
}
