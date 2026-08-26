use foundation::model::ModelManifest;
use runtime::{
    Result as RuntimeResult, backend::ModelHandle, kv::CacheConfig, progress::ProgressEvent,
};

use super::{Engine, EngineInner};

impl Engine {
    /// Loads model weights and reports backend progress events.
    pub fn load_model_with_progress(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        self.load_model_with_progress_and_reservation(manifest, None, None, progress)
    }

    pub(crate) fn load_model_with_progress_and_reservation(
        &self,
        manifest: &ModelManifest,
        reserved_bytes: Option<usize>,
        cache: Option<CacheConfig>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&manifest, &mut *progress);
        #[cfg(not(feature = "metal"))]
        let _ = (&reserved_bytes, &cache);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.load_model_with_progress(manifest, progress)?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let mut mapped = |event| progress(super::metal_progress(event));
                match (reserved_bytes, cache) {
                    (Some(bytes), Some(cache)) => metal.load_model_with_progress_and_reservation(
                        manifest, bytes, cache, &mut mapped,
                    ),
                    _ => metal.load_model_with_progress(manifest, &mut mapped),
                }
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }
}
