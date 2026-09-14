use super::{Engine, EngineInner};
use crate::Result;

impl Engine {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(clippy::unnecessary_wraps, reason = "CUDA performs a fallible model lookup")
    )]
    pub(crate) fn prefill_profile_shapes(
        &self,
        model: &runtime::backend::ModelHandle,
        max_batch_tokens: usize,
    ) -> Result<Vec<usize>> {
        #[cfg(not(feature = "cuda"))]
        let _ = (model, max_batch_tokens);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                Ok(cuda.prefill_profile_shapes(&model.id, max_batch_tokens)?)
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Ok(Vec::new()),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Ok(Vec::new()),
        }
    }

    /// Enables or disables backend decode profiling where supported.
    pub fn set_profile_decode(&self, enabled: bool) -> Result<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = enabled;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.set_profile_decode(enabled)?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                metal.set_profile_decode(enabled);
                Ok(())
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Ok(()),
        }
    }

    /// Starts an accelerator-profiler capture range without changing execution
    /// policy.
    pub fn start_profiler_capture(&self) -> Result<()> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.start_profiler_capture()?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Ok(()),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Ok(()),
        }
    }

    /// Stops the active accelerator-profiler capture range.
    pub fn stop_profiler_capture(&self) -> Result<()> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.stop_profiler_capture()?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Ok(()),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Ok(()),
        }
    }
}
