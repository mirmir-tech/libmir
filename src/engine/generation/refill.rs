use runtime::{backend::PrefillRequest, progress::ProgressEvent};

use super::{Engine, EngineInner, EnginePrefillBatch};
use crate::Result;

impl Engine {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(clippy::unnecessary_wraps, reason = "CUDA refill is fallible")
    )]
    pub(crate) fn extend_generation_prefill(
        &self,
        batch: &mut EnginePrefillBatch,
        requests: &[PrefillRequest],
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<bool> {
        match (&self.inner, batch) {
            #[cfg(feature = "cuda")]
            (EngineInner::Cuda(cuda), EnginePrefillBatch::Cuda(batch)) => {
                cuda.extend_prefill_batch(batch, requests, progress)?;
                Ok(true)
            },
            #[cfg(feature = "metal")]
            (EngineInner::Metal(_), EnginePrefillBatch::Metal(_)) => {
                let _ = (requests, progress);
                Ok(false)
            },
            #[cfg(all(feature = "cuda", feature = "metal"))]
            _ => Ok(false),
        }
    }
}
