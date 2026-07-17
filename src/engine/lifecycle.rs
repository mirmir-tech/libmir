#[cfg(any(feature = "cuda", feature = "metal"))]
use runtime::RuntimeError;
use runtime::{Result as RuntimeResult, backend::ModelHandle};

use super::Engine;
#[cfg(any(feature = "cuda", feature = "metal"))]
use super::EngineInner;

impl Engine {
    pub(crate) fn unload_model(&self, model: &ModelHandle) -> RuntimeResult<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let _ = (self, model);
            super::unavailable()
        }
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            let unloaded = match &self.inner {
                #[cfg(feature = "cuda")]
                EngineInner::Cuda(cuda) => cuda.unload_model(&model.id)?,
                #[cfg(feature = "metal")]
                EngineInner::Metal(metal) => metal.unload_model(model)?,
            };
            if !unloaded {
                return Err(RuntimeError::ModelNotLoaded(model.id.clone()));
            }
            Ok(())
        }
    }
}
