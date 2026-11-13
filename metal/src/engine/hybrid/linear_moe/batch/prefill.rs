use super::{Array, DecoderCache, HybridLinearMoeModel, Result, Stream};
use crate::engine::decoder::{
    LoweredLayer,
    packed_profile::{self, Component, PrefillProfile},
};

impl HybridLinearMoeModel {
    pub(crate) fn forward_packed_prefill_state(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        let shape = token_ids.shape()?;
        let position = positions.iter().copied().min().map_or(Ok(0), usize::try_from)?;
        let evaluation_step = crate::engine::decoder::prefill_evaluation_step(
            caches.len(),
            usize::try_from(shape[1])?,
            position,
            self.layers.len(),
        );
        let profile = PrefillProfile::begin(&hidden, caches, positions, stream)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let started = profile.start();
            hidden = layer.mix_packed_prefill(&hidden, caches, positions, stream)?;
            packed_profile::record(
                started,
                &hidden,
                caches,
                stream,
                index,
                Component::Mixer(layer.mixer_kind()),
            )?;
            let started = profile.start();
            #[cfg(test)]
            if matches!(
                stream.config().diagnostics.moe_prefill,
                crate::config::MoePrefill::CompareRoutes
                    | crate::config::MoePrefill::CompareProjection
                    | crate::config::MoePrefill::CompareIndexed
                    | crate::config::MoePrefill::MeasureIndexedNumerics
                    | crate::config::MoePrefill::CompareAligned
            ) && index == 0
                && positions.iter().all(|&position| position == 512)
            {
                let input = layer.post_attention_norm.apply(&hidden, layer.rms_norm_eps, stream)?;
                layer.moe.diagnose_prefill(&input, stream)?;
            }
            #[cfg(test)]
            if matches!(
                stream.config().diagnostics.moe_prefill,
                crate::config::MoePrefill::CompareAlignmentBudgets
                    | crate::config::MoePrefill::CompareTileWidths
                    | crate::config::MoePrefill::CompareTiles
                    | crate::config::MoePrefill::ProfileTiles
            ) && [0, self.layers.len() / 2, self.layers.len() - 1].contains(&index)
                && positions.iter().all(|&position| position == 512)
            {
                let input = layer.post_attention_norm.apply(&hidden, layer.rms_norm_eps, stream)?;
                layer.moe.compare_alignment_budgets(&input, index, stream)?;
            }
            hidden = layer.feed_forward_packed_prefill(&hidden, stream)?;
            packed_profile::record(
                started,
                &hidden,
                caches,
                stream,
                index,
                Component::FeedForward,
            )?;
            if evaluation_step
                .is_some_and(|step| (index + 1) % step == 0 || index + 1 == self.layers.len())
            {
                crate::engine::diagnostics::evaluate_segment(&hidden, stream)?;
            }
        }
        self.final_norm.apply(&hidden, self.rms_norm_eps, stream)
    }
}

#[cfg(test)]
impl super::HybridLinearMoeLayer {
    pub(super) fn try_packed_linear_prefill(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Option<Array>> {
        use crate::config::GatedDeltaPrefill;
        let mode = stream.config().diagnostics.gated_delta_prefill;
        let super::Attention::Linear(layer) = &self.attention else {
            return Ok(None);
        };
        if mode == GatedDeltaPrefill::Rows {
            return Ok(None);
        }
        let mut states = caches
            .iter_mut()
            .map(|cache| cache.gated_delta_state(self.index))
            .collect::<Result<Vec<_>>>()?;
        if mode == GatedDeltaPrefill::ComparePacked {
            if self.index == 0 && positions.iter().all(|&position| position == 512) {
                layer.compare_packed_prefill(input, &states, stream)?;
            }
            return Ok(None);
        }
        if self.index == 0
            && positions.iter().all(|&position| position == 0)
            && mode == GatedDeltaPrefill::DiagnosePacked
        {
            layer.diagnose_packed_projection(input, stream)?;
        }
        layer.forward_packed_prefill(input, &mut states, stream)
    }
}
