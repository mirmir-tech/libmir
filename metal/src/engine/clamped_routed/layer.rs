use models::weights::RoutedDecoderLayerBindings;

use super::{
    attention::ClampedRoutedAttention, config::ClampedRoutedConfig, experts::ClampedRoutedExperts,
};
use crate::engine::{
    Array, DecoderCache, Error, ModelTensors, NormWeight, Result, Stream,
    decoder::{LayerContext, LoweredLayer, MixerKind},
};

#[derive(Debug)]
pub(super) struct ClampedRoutedLayer {
    attention: ClampedRoutedAttention,
    moe_norm: NormWeight,
    experts: ClampedRoutedExperts,
    epsilon: f32,
}

impl ClampedRoutedLayer {
    pub fn load(
        tensors: &ModelTensors,
        bindings: RoutedDecoderLayerBindings<'_>,
        config: ClampedRoutedConfig,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            attention: ClampedRoutedAttention::load(tensors, bindings, config, stream)?,
            moe_norm: NormWeight::load_name(tensors, &bindings.post_attention_norm.source)?,
            experts: ClampedRoutedExperts::load(tensors, bindings, config, stream)?,
            epsilon: config.epsilon,
        })
    }
}

impl LoweredLayer for ClampedRoutedLayer {
    fn mixer_kind(&self) -> MixerKind {
        MixerKind::Softmax
    }

    fn forward_mixer(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        index: usize,
        context: LayerContext<'_>,
    ) -> Result<Array> {
        let cache = cache
            .attention_caches_mut()?
            .get_mut(index)
            .ok_or_else(|| Error::InvalidModel(format!("missing cache for layer {index}")))?;
        let attention = self
            .attention
            .forward(input, cache, context.position, context.causal, context.stream)?;
        input.add(&attention, context.stream)
    }

    fn forward_feed_forward(&self, input: &Array, context: LayerContext<'_>) -> Result<Array> {
        let normalized = self.moe_norm.apply(input, self.epsilon, context.stream)?;
        input.add(&self.experts.forward(&normalized, context.stream)?, context.stream)
    }
}
