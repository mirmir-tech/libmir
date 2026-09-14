use super::{ClampedRoutedModel, layer::ClampedRoutedLayer};
use crate::engine::{
    Array, DecoderCache, Error, Result, Stream, decode_graph,
    decoder::{
        LayerContext, LoweredLayer, LoweredPackedLayer, forward_packed_layers,
        packed_profile::{self, Component, PrefillProfile},
        prefill_evaluation_step,
    },
};

impl ClampedRoutedModel {
    pub fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.forward_packed_state(token_ids, caches, positions, false, stream)?;
        let hidden = self.final_norm.apply(&hidden, self.config.epsilon, stream)?;
        let logits = self.output.forward(&hidden, stream)?;
        decode_graph::export_once(&logits, stream)?;
        Ok(logits)
    }

    pub fn forward_packed_state(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = token_ids.shape()?;
        if shape.len() != 2
            || caches.is_empty()
            || usize::try_from(shape[0])? != caches.len()
            || positions.len() != caches.len()
            || positions.iter().any(|position| *position < 0)
        {
            return Err(Error::InvalidModel("invalid packed clamped token batch".into()));
        }
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        if !causal {
            return forward_packed_layers(&self.layers, hidden, caches, positions, stream);
        }
        let evaluation_step = if causal {
            prefill_evaluation_step(
                caches.len(),
                usize::try_from(shape[1])?,
                usize::try_from(positions.iter().copied().max().unwrap_or_default())?,
                self.layers.len(),
            )
        } else {
            None
        };
        let profile = PrefillProfile::begin(&hidden, caches, positions, stream)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let started = profile.start();
            let mixed = layer
                .attention
                .forward_packed(&hidden, caches, positions, index, causal, stream)?;
            hidden = hidden.add(&mixed, stream)?;
            packed_profile::record(
                started,
                &hidden,
                caches,
                stream,
                index,
                Component::Mixer(layer.mixer_kind()),
            )?;
            let started = profile.start();
            hidden = layer.forward_feed_forward(
                &hidden,
                LayerContext {
                    position: 0,
                    causal,
                    positions: None,
                    image: None,
                    stream,
                },
            )?;
            packed_profile::record(
                started,
                &hidden,
                caches,
                stream,
                index,
                Component::FeedForward,
            )?;
            if evaluation_step.is_some_and(|step| (index + 1) % step == 0) {
                let mut roots = vec![&hidden];
                for cache in caches.iter() {
                    cache.extend_graph_roots(&mut roots);
                }
                crate::engine::diagnostics::measure(
                    crate::engine::diagnostics::Stage::Segment,
                    || stream.eval_many_with_paged_arenas(&roots),
                )?;
            }
        }
        Ok(hidden)
    }
}

impl LoweredPackedLayer for ClampedRoutedLayer {
    fn forward_packed_mixer(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        index: usize,
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mixed =
            self.attention.forward_packed(input, caches, positions, index, false, stream)?;
        input.add(&mixed, stream)
    }

    fn forward_packed_feed_forward(
        &self,
        input: &Array,
        _batch_size: usize,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_feed_forward(
            input,
            LayerContext {
                position: 0,
                causal: false,
                positions: None,
                image: None,
                stream,
            },
        )
    }
}
