use runtime::Result;

use crate::native::{
    backend::MetalBackend,
    prefill::{MetalPrefillBatch, MetalPrefillCohort},
};

impl MetalBackend {
    /// Retires selected rows at a generation-step boundary, preserving other
    /// rows.
    pub fn cancel_prefill_sessions(
        &self,
        batch: &MetalPrefillBatch,
        sessions: &[uuid::Uuid],
    ) -> Result<()> {
        let model = batch.model_id().to_owned();
        let batch = batch.clone();
        let sessions = sessions.to_vec();
        Ok(self.with_model(&model, move |loaded| batch.cancel_sessions(loaded, &sessions))?)
    }

    /// Releases prefix leases belonging to requests that never entered a wave.
    pub fn discard_prefill_cohort_sessions(
        &self,
        cohort: &MetalPrefillCohort,
        sessions: &[uuid::Uuid],
    ) -> Result<()> {
        let model = cohort.model_id().to_owned();
        let cohort = cohort.clone();
        let sessions = sessions.to_vec();
        Ok(self.with_model(&model, move |_| cohort.discard(&sessions))?)
    }
}
