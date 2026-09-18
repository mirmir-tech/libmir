use runtime::backend::PrefillOutput;

use super::{Engine, EngineInner, EnginePrefillBatch};
use crate::Result;

impl Engine {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(clippy::unnecessary_wraps, reason = "only CUDA completes fallible partial prefills")
    )]
    pub(crate) fn take_completed_generation_prefill(
        &self,
        batch: &mut EnginePrefillBatch,
    ) -> Result<Vec<(uuid::Uuid, PrefillOutput)>> {
        match (&self.inner, batch) {
            #[cfg(feature = "cuda")]
            (EngineInner::Cuda(backend), EnginePrefillBatch::Cuda(batch)) => {
                Ok(backend.take_completed_prefill_rows(batch)?)
            },
            #[cfg(feature = "metal")]
            (EngineInner::Metal(_), EnginePrefillBatch::Metal(_)) => Ok(Vec::new()),
            #[cfg(all(feature = "metal", feature = "cuda"))]
            _ => Err(super::batch_backend_mismatch()),
        }
    }
}
