use super::HybridMoeLayer;
use crate::engine::{
    Array, DecoderCache, Error, Result,
    decoder::{LayerContext, LoweredLayer, MixerKind},
    prefix_attention_mask,
};

impl LoweredLayer for HybridMoeLayer {
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
        let mask = context
            .image
            .filter(|_| self.config.max_context.is_some())
            .map(|image| {
                let sequence = input.shape()?.get(1).copied().ok_or(Error::ShapeOverflow)?;
                prefix_attention_mask(
                    usize::try_from(sequence)?,
                    image,
                    self.config.max_context,
                    context.stream,
                )
            })
            .transpose()?;
        let cache = cache
            .attention_caches_mut()?
            .get_mut(index)
            .ok_or_else(|| Error::InvalidModel(format!("missing cache for layer {index}")))?;
        self.mix(
            input,
            Some(cache),
            context.position,
            context.causal,
            mask.as_ref(),
            context.stream,
        )
    }

    fn forward_feed_forward(&self, input: &Array, context: LayerContext<'_>) -> Result<Array> {
        self.feed_forward(input, context.causal, context.stream)
    }
}
