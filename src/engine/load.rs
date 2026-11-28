use architecture::InspectedModel;
use foundation::model::ModelManifest;
use runtime::{
    Result as RuntimeResult, backend::ModelHandle, kv::CacheConfig, progress::ProgressEvent,
};

use super::{Engine, EngineInner};

/// Backend-specific architecture admitted once before loading weights.
pub enum AdmittedArchitecture {
    #[cfg(feature = "cuda")]
    Cuda(architecture::cuda::CudaArchitecture),
    #[cfg(feature = "metal")]
    Metal(architecture::metal::MetalArchitecture),
}

impl AdmittedArchitecture {
    pub(crate) const fn resizable_kv_cache(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(admitted) => admitted.resizable_kv_cache(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => false,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            _ => unreachable!(),
        }
    }
}

impl Engine {
    /// Loads model weights and reports backend progress events.
    pub fn load_model_with_progress(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        let inspected = InspectedModel::inspect_directory(&manifest.path)
            .map_err(|error| runtime::RuntimeError::Backend(error.to_string()))?;
        let admitted = self
            .admit(&inspected)
            .map_err(|error| runtime::RuntimeError::Backend(error.to_string()))?;
        self.load_admitted(manifest, &inspected, admitted, None, None, None, progress)
    }

    pub(crate) fn admit(
        &self,
        inspected: &InspectedModel,
    ) -> architecture::Result<AdmittedArchitecture> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = inspected;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => inspected.admit_cuda().map(AdmittedArchitecture::Cuda),
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => inspected.admit_metal().map(AdmittedArchitecture::Metal),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable(never) => match *never {},
        }
    }

    /// Loads an already-inspected, already-admitted checkpoint onto the active
    /// accelerator without re-reading layout files or re-running admission.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_admitted(
        &self,
        manifest: &ModelManifest,
        inspected: &InspectedModel,
        admitted: AdmittedArchitecture,
        reserved_bytes: Option<usize>,
        cache: Option<CacheConfig>,
        sequence_capacity_blocks: Option<u32>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (manifest, inspected, &mut *progress);
        #[cfg(not(feature = "metal"))]
        let _ = (&reserved_bytes, &cache);
        #[cfg(not(feature = "cuda"))]
        let _ = sequence_capacity_blocks;
        match (&self.inner, admitted) {
            #[cfg(feature = "cuda")]
            (EngineInner::Cuda(cuda), AdmittedArchitecture::Cuda(admitted)) => Ok(cuda
                .load_admitted(
                    manifest,
                    inspected,
                    &admitted,
                    sequence_capacity_blocks,
                    progress,
                )?),
            #[cfg(feature = "metal")]
            (EngineInner::Metal(metal), AdmittedArchitecture::Metal(admitted)) => {
                let reservation = reserved_bytes.zip(cache);
                metal.load_admitted(manifest, inspected.clone(), admitted, reservation, progress)
            },
            #[cfg(all(feature = "cuda", feature = "metal"))]
            _ => Err(runtime::RuntimeError::Backend(
                "admitted architecture does not match the active accelerator engine".into(),
            )),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            (EngineInner::Unavailable(never), _) => match *never {},
        }
    }
}
