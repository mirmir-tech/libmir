use super::{Error, LoadedModel, NativePrefill, Result, Sequence, evaluation};
use crate::native::prefill::diagnostics;

impl Sequence {
    pub(super) fn complete(
        &mut self,
        loaded: &mut LoadedModel,
        logits: crate::engine::Array,
    ) -> Result<()> {
        let tokens = &self.request.prompt_tokens;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?;
        evaluation::materialize(loaded, state, &logits)?;
        state.position = tokens.len();
        let prefix_bytes = loaded
            .estimated_prefix_bytes(tokens.len())?
            .checked_add(logits.byte_len()?)
            .ok_or(crate::engine::Error::ShapeOverflow)?;
        let _cached = crate::native::model::cache_prefix_snapshot(
            &mut loaded.prefixes,
            &loaded.info.manifest.id,
            tokens,
            state,
            &logits,
            self.request.block_table.block_size(),
            prefix_bytes,
        )?;
        let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
        let model = loaded.execution.decoder()?;
        let output = diagnostics::measure(diagnostics::Stage::FirstToken, || {
            crate::native::prefill::reservation::output(
                model,
                &loaded.stream,
                state,
                logits,
                self.execution_sampling,
            )
        })?;
        let state = self
            .state
            .take()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?;
        loaded.sessions.insert(self.request.session_id, state);
        self.position = tokens.len();
        self.page_reservation_pending = false;
        self.output = Some(NativePrefill {
            output,
            prefix_cache_tokens: self.prefix_cache_tokens,
        });
        Ok(())
    }
}
