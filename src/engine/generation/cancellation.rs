use super::{Engine, EngineInner, EnginePrefillBatch, EnginePrefillCohort};
use crate::Result;

impl Engine {
    pub(crate) fn cancel_generation_prefill(
        &self,
        batch: &mut EnginePrefillBatch,
        sessions: &[uuid::Uuid],
    ) -> Result<()> {
        match (&self.inner, batch) {
            #[cfg(feature = "metal")]
            (EngineInner::Metal(backend), EnginePrefillBatch::Metal(batch)) => {
                Ok(backend.cancel_prefill_sessions(batch, sessions)?)
            },
            #[cfg(feature = "cuda")]
            (EngineInner::Cuda(backend), EnginePrefillBatch::Cuda(batch)) => {
                Ok(backend.cancel_prefill_sessions(batch, sessions)?)
            },
            #[cfg(all(feature = "metal", feature = "cuda"))]
            _ => Err(super::batch_backend_mismatch()),
        }
    }

    #[cfg_attr(
        not(feature = "metal"),
        expect(
            clippy::unnecessary_wraps,
            reason = "Metal cohort leases have fallible worker ownership"
        )
    )]
    pub(crate) fn discard_prefill_cohort_sessions(
        &self,
        cohort: &EnginePrefillCohort,
        sessions: &[uuid::Uuid],
    ) -> Result<()> {
        #[cfg(feature = "metal")]
        if let (EngineInner::Metal(backend), Some(cohort)) = (&self.inner, &cohort.metal) {
            backend.discard_prefill_cohort_sessions(cohort, sessions)?;
        }
        #[cfg(not(feature = "metal"))]
        let _ = (self, cohort, sessions);
        Ok(())
    }
}
