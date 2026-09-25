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
        self.load_model_with_progress_and_reservation(manifest, None, None, None, progress)
    }

    /// Whether `manifest` loads into a runtime that can reallocate its K/V
    /// pages afterwards, so the cache may start provisional and be measured.
    pub(crate) fn kv_cache_resizable(&self, manifest: &ModelManifest) -> RuntimeResult<bool> {
        #[cfg(not(feature = "cuda"))]
        let _ = manifest;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => Ok(cuda::CudaEngine::kv_cache_resizable(manifest)?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Ok(false),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }

    /// `sequence_capacity_blocks` bounds one sequence's block table where the
    /// cache loaded now is provisional and grows after warm-up.
    pub(crate) fn load_model_with_progress_and_reservation(
        &self,
        manifest: &ModelManifest,
        reserved_bytes: Option<usize>,
        cache: Option<CacheConfig>,
        sequence_capacity_blocks: Option<u32>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&manifest, &mut *progress);
        #[cfg(not(feature = "metal"))]
        let _ = (&reserved_bytes, &cache);
        #[cfg(not(feature = "cuda"))]
        let _ = sequence_capacity_blocks;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.load_model_with_sequence_capacity(
                manifest,
                sequence_capacity_blocks,
                progress,
            )?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => match (reserved_bytes, cache) {
                (Some(bytes), Some(cache)) => {
                    metal.load_model_with_progress_and_reservation(manifest, bytes, cache, progress)
                },
                _ => metal.load_model_with_progress(manifest, progress),
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }
}
