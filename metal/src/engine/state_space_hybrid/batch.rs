//! Packed decode and prefill: rows are independent; attention runs per row,
//! while Mamba-2 state and routed experts batch over the whole row set.

use super::{layer::StateSpaceHybridLayer, model::StateSpaceHybridModel};
use crate::engine::{
    Array, DecoderCache, Result, Stream, decode,
    decoder::{
        LoweredLayer, LoweredPackedLayer, forward_packed_layers,
        packed_profile::{self, Component, PrefillProfile},
        prefill_evaluation_step,
    },
};

impl StateSpaceHybridModel {
    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        let hidden = forward_packed_layers(&self.layers, hidden, caches, positions, stream)?;
        let hidden = self.final_norm.apply(&hidden, self.epsilon, stream)?;
        let logits = self.project_output(&hidden, stream)?;
        decode::graph::export_once(&logits, stream)?;
        Ok(logits)
    }

    pub(crate) fn forward_packed_prefill_state(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let shape = token_ids.shape()?;
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        let position = positions.iter().copied().max().map_or(Ok(0), usize::try_from)?;
        let evaluation_step = prefill_evaluation_step(
            caches.len(),
            usize::try_from(shape[1])?,
            position,
            self.layers.len(),
        );
        let profile = PrefillProfile::begin(&hidden, caches, positions, stream)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let started = profile.start();
            hidden = layer.mix_packed(&hidden, caches, true, stream)?;
            packed_profile::record(
                started,
                &hidden,
                caches,
                stream,
                index,
                Component::Mixer(layer.mixer_kind()),
            )?;
            let started = profile.start();
            hidden = layer.feed_forward(&hidden, stream)?;
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
        self.final_norm.apply(&hidden, self.epsilon, stream)
    }
}

impl LoweredPackedLayer for StateSpaceHybridLayer {
    fn forward_packed_mixer(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        _index: usize,
        _positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.mix_packed(input, caches, false, stream)
    }

    fn forward_packed_feed_forward(
        &self,
        input: &Array,
        _batch_size: usize,
        stream: &Stream,
    ) -> Result<Array> {
        self.feed_forward(input, stream)
    }
}
