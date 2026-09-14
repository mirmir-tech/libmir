use super::{
    HybridMixerBindings, SharedRoutedFeedForwardBindings, feed_forward, layer, mixer, optional,
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
        let feed_forward = if optional(self, index, LayerTensorRole::Router).is_some() {
            MixedFeedForwardBindings::SharedRouted(feed_forward(self, index)?)
        } else {
            let get = |projection| {
                layer(self, index, LayerTensorRole::FeedForwardProjection { projection })
            };
            MixedFeedForwardBindings::Dense(DenseFeedForwardBindings {
                gate: get(FeedForwardProjectionRole::Gate)?,
                up: get(FeedForwardProjectionRole::Up)?,
                down: get(FeedForwardProjectionRole::Down)?,
            })
        };
        Ok(MixedDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            mixer: mixer(self, index)?,
            post_attention_norm: layer(self, index, LayerTensorRole::PostAttentionNorm)?,
            feed_forward,
        })
    }
}

impl<'a> MixedDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = vec![self.input_norm, self.post_attention_norm];
        bindings.extend(self.mixer.bindings());
        match self.feed_forward {
            MixedFeedForwardBindings::Dense(value) => {
                bindings.extend([value.gate, value.up, value.down]);
            },
            MixedFeedForwardBindings::SharedRouted(value) => bindings.extend(value.bindings()),
        }
        super::sources(bindings)
    }
}
