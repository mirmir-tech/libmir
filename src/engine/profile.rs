use super::{Engine, EngineInner};
use crate::Result;

impl Engine {
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
