use super::{
    super::slot::LayerSlot, HybridMixerBindings, SharedRoutedFeedForwardBindings, feed_forward,
    layer, mixer, optional,
};
use crate::{
    error::Result,
    weights::{FeedForwardProjectionRole, LayerTensorRole, TensorBinding, WeightBindingPlan},
};

#[derive(Debug, Clone, Copy)]
pub struct DenseFeedForwardBindings<'a> {
    pub gate: &'a TensorBinding,
    pub up: &'a TensorBinding,
    pub down: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub enum MixedFeedForwardBindings<'a> {
    Dense(DenseFeedForwardBindings<'a>),
    SharedRouted(SharedRoutedFeedForwardBindings<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct MixedDecoderLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub mixer: HybridMixerBindings<'a>,
    pub post_attention_norm: &'a TensorBinding,
    pub feed_forward: MixedFeedForwardBindings<'a>,
}

impl WeightBindingPlan {
    pub fn mixed_decoder_layer(&self, index: usize) -> Result<MixedDecoderLayerBindings<'_>> {
        self.mixed_layer(LayerSlot::Decoder(index))
    }

    /// The decoder layer of a decoder slot or a prediction module.
    pub(in crate::weights::binding) fn mixed_layer(
        &self,
        slot: LayerSlot,
    ) -> Result<MixedDecoderLayerBindings<'_>> {
        Ok(MixedDecoderLayerBindings {
            input_norm: layer(self, slot, LayerTensorRole::InputNorm)?,
            mixer: mixer(self, slot)?,
            post_attention_norm: layer(self, slot, LayerTensorRole::PostAttentionNorm)?,
            feed_forward: mixed_feed_forward(self, slot)?,
        })
    }
}

/// A routed feed-forward with shared experts where the layer has a router,
/// a dense one otherwise.
pub(super) fn mixed_feed_forward(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
) -> Result<MixedFeedForwardBindings<'_>> {
    if optional(plan, slot, LayerTensorRole::Router).is_some() {
        return feed_forward(plan, slot).map(MixedFeedForwardBindings::SharedRouted);
    }
    let get = |projection| layer(plan, slot, LayerTensorRole::FeedForwardProjection { projection });
    Ok(MixedFeedForwardBindings::Dense(DenseFeedForwardBindings {
        gate: get(FeedForwardProjectionRole::Gate)?,
        up: get(FeedForwardProjectionRole::Up)?,
        down: get(FeedForwardProjectionRole::Down)?,
    }))
}

impl<'a> MixedFeedForwardBindings<'a> {
    pub(super) fn bindings(self) -> Vec<&'a TensorBinding> {
        match self {
            Self::Dense(value) => vec![value.gate, value.up, value.down],
            Self::SharedRouted(value) => value.bindings(),
        }
    }
}

impl<'a> MixedDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = vec![self.input_norm, self.post_attention_norm];
        bindings.extend(self.mixer.bindings());
        bindings.extend(self.feed_forward.bindings());
        super::sources(bindings)
    }
}
