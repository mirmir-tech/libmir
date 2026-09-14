use super::{LoadedModel, clear_memory_cache, pressure::prefix_reclamation_needed};
use crate::{
    engine::{MemoryStats, persistent_history::budget::Admission},
    native::error::Result,
};

impl LoadedModel {
    // Called before prefix leasing/eviction for both scalar and packed prefill.
    // Keep optional duplicate histories suspended until a later prefill boundary
    // observes relief, rather than rebuilding them immediately under pressure.
    pub(in crate::native) fn apply_history_pressure(&mut self, memory: MemoryStats) -> Result<()> {
        if !prefix_reclamation_needed(memory) {
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
