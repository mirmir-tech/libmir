use super::{
    LoadedModel, clear_memory_cache,
    pressure::{history_limit_bytes, history_pressure},
};
use crate::{
    engine::{MemoryStats, persistent_history::budget::Admission},
    native::error::Result,
};

impl LoadedModel {
    /// Gives the history back the room held for a prefill once no prefill
    /// batch is in flight: rows that decode after their whole cohort has
    /// prefilled compete with no prompt still to come.
    pub(in crate::native) fn release_settled_prefill_reserve(&mut self) -> Result<()> {
        if !self.history_prefill_reserved || self.prefill_reservations.in_flight() {
            return Ok(());
        }
        self.history_prefill_reserved = false;
        let memory = crate::engine::memory_stats()?;
        let retained = self.stream.history_budget().snapshot().retained_bytes;
        self.stream
            .history_budget()
            .set_automatic_limit(history_limit_bytes(memory, retained, 0));
        if !history_pressure(memory, 0) {
            self.stream.history_budget().set_admission(Admission::Open);
        }
        Ok(())
    }

    // Called before prefix leasing/eviction for both scalar and packed prefill.
    // Keep optional duplicate histories suspended until a later prefill boundary
    // observes relief, rather than rebuilding them immediately under pressure.
    //
    /// Keeps room for `prefill_tokens` of K/V that the starting prefill will
    /// still allocate: rows that finish early decode beside rows still
    /// prefilling.
    pub(in crate::native) fn apply_history_pressure(
        &mut self,
        memory: MemoryStats,
        prefill_tokens: usize,
    ) -> Result<()> {
        // Arenas keep their high-water capacity: only prompt tokens beyond the
        // pages they already hold allocate memory.
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        let covered = self.allocated_page_headroom()?.saturating_mul(page_size);
        let pending = prefill_tokens
            .saturating_sub(covered)
            .saturating_mul(self.kv_rate_bytes_per_token()?);
        let retained = self.stream.history_budget().snapshot().retained_bytes;
        let limit = history_limit_bytes(memory, retained, pending);
        tracing::debug!(
            active_bytes = memory.active,
            cached_bytes = memory.cached,
            retained_history_bytes = retained,
            prefill_tokens,
            covered_tokens = covered,
            pending_bytes = pending,
            history_limit_bytes = limit,
            pressure = history_pressure(memory, pending),
            "applied Metal history pressure before prefill"
        );
        self.stream.history_budget().set_automatic_limit(limit);
        self.history_prefill_reserved = pending > 0;
        if !history_pressure(memory, pending) {
            self.stream.history_budget().set_admission(Admission::Open);
            return Ok(());
        }
        self.stream.history_budget().set_admission(Admission::Suspended);
        let before = self.stream.history_budget().snapshot().retained_bytes;
        if before == 0 {
            return Ok(());
        }
        for state in self.sessions.values_mut() {
            state.cache.release_history();
        }
        clear_memory_cache()?;
        let after = self.stream.history_budget().snapshot().retained_bytes;
        tracing::debug!(
            retained_history_bytes_before = before,
            retained_history_bytes_after = after,
            "released optional batch histories before prefix reclamation"
        );
        Ok(())
    }
}
