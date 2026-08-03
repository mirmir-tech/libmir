mod backend;
mod batch;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod generation;
mod lifecycle;
mod load;
mod memory;
mod prefill;
mod select;
mod tasks;
mod telemetry;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod vision;

#[cfg(feature = "cuda")]
use cuda::CudaEngine;
use foundation::model::BackendTarget;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use generation::PrefillExecutionProfile;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use generation::{EngineGenerationStepOutput, EnginePrefillBatch};
#[cfg(feature = "metal")]
use metal::MetalBackend;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use runtime::RuntimeError;
#[cfg(feature = "cuda")]
use runtime::backend::DecodeRequest;
use runtime::{
    Result as RuntimeResult,
    backend::{DecodeOutput, ModelHandle, SamplingLogits},
    kv::BlockTable,
    progress::ProgressEvent,
};
use uuid::Uuid;

use crate::{Result, RuntimeConfig};

#[derive(Debug, Clone)]
/// Backend-neutral accelerator engine selected from [`RuntimeConfig`].
pub struct Engine {
    inner: EngineInner,
}

#[derive(Debug, Clone)]
enum EngineInner {
    #[cfg(feature = "cuda")]
    #[cfg_attr(
        all(feature = "metal", target_os = "macos"),
        expect(dead_code, reason = "all-features builds select Metal on macOS")
    )]
    Cuda(CudaEngine),
    #[cfg(feature = "metal")]
    Metal(MetalBackend),
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    Unavailable,
}

impl Engine {
    /// Initializes the accelerator selected by the enabled Cargo feature.
    pub fn from_config(config: &RuntimeConfig) -> Result<Self> {
        select::engine(config)
    }

    #[must_use]
    /// Returns the backend targeted by this engine.
    pub const fn target(&self) -> BackendTarget {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => BackendTarget::Cuda,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => BackendTarget::Metal,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => BackendTarget::CpuReference,
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

    #[cfg_attr(
        not(feature = "metal"),
        expect(
            clippy::unnecessary_wraps,
            reason = "Metal and CUDA builds share one startup-tuning interface"
        )
    )]
    pub(crate) fn finish_startup_tuning(&self, model: &ModelHandle) -> Result<()> {
        #[cfg(not(feature = "metal"))]
        let _ = model;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                cuda.finish_startup_tuning();
                Ok(())
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => Ok(metal.finish_startup_tuning(model)?),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Ok(()),
        }
    }

    /// Decodes one token for a session and returns the next-token prediction.
    pub fn decode_token(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        token_id: u32,
        block_table: &BlockTable,
        sampling: SamplingLogits,
    ) -> RuntimeResult<DecodeOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&model, session_id, token_id, &block_table, sampling);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.decode_token(&DecodeRequest {
                model: model.clone(),
                session_id,
                token_id,
                block_table: block_table.clone(),
                sampling_logits: sampling,
            })?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                metal.decode_token(model, session_id, token_id, block_table, sampling)
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    /// Releases allocations retained in the backend memory cache.
    pub fn clear_memory_cache(&self) -> RuntimeResult<()> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.clear_memory_cache()?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.clear_memory_cache(),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    /// Clears cached prefixes and sessions associated with `model`.
    pub fn clear_prefix_cache(&self, model: &ModelHandle) -> RuntimeResult<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = model;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.clear_model_sessions(&model.id)?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.clear_prefix_cache(model),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    /// Releases backend state owned by one session.
    pub fn release_session(&self, model: &ModelHandle, session: Uuid) -> RuntimeResult<()> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (model, session);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.release_session(&model.id, session)?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.release_session(model, session),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    /// Runs a closure inside a backend-specific generation resource scope.
    pub fn with_generation_scope<T>(&self, run: impl FnOnce() -> T) -> T {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => run(),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.with_generation_scope(run),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => run(),
        }
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn unavailable<T>() -> RuntimeResult<T> {
    Err(RuntimeError::BackendUnavailable(
        "libmir was built without an accelerator feature".into(),
    ))
}

#[cfg(feature = "metal")]
fn metal_progress(event: metal::MetalProgressEvent) -> ProgressEvent {
    ProgressEvent {
        stage: match event.stage {
            metal::MetalProgressStage::LoadWeights => runtime::progress::ProgressStage::LoadWeights,
            metal::MetalProgressStage::PrefillTokens => {
                runtime::progress::ProgressStage::PrefillTokens
            },
        },
        current: event.current,
        total: event.total,
        unit: match event.unit {
            metal::MetalProgressUnit::Byte => runtime::progress::ProgressUnit::Byte,
            metal::MetalProgressUnit::Token => runtime::progress::ProgressUnit::Token,
        },
        detail: event.detail,
    }
}
