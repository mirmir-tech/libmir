mod backend;
mod batch;
mod lifecycle;
mod memory;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod pooled_vision;
mod select;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod spatial_merge_vision;

#[cfg(feature = "cuda")]
use cuda::CudaEngine;
use foundation::model::{BackendTarget, ModelManifest};
#[cfg(feature = "metal")]
use metal::MetalBackend;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use runtime::RuntimeError;
#[cfg(feature = "cuda")]
use runtime::backend::{DecodeRequest, PrefillRequest};
use runtime::{
    Result as RuntimeResult,
    backend::{DecodeOutput, ModelHandle, PrefillOutput, SamplingLogits},
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
    pub fn set_profile_decode(&self, enabled: bool) {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = enabled;
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        let _ = enabled;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => {},
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.set_profile_decode(enabled),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => {},
        }
    }

    /// Loads model weights and reports backend progress events.
    pub fn load_model_with_progress(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&manifest, &progress);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.load_model_with_progress(manifest, progress)?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let mut mapped = |event| progress(metal_progress(event));
                metal.load_model_with_progress(manifest, &mut mapped)
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Prefills one session from token identifiers and its allocated cache
    /// blocks.
    pub fn prefill_tokens_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt_tokens: &[u32],
        block_table: &BlockTable,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&model, session_id, &prompt_tokens, &block_table, sampling, &progress);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.prefill_with_progress(
                &PrefillRequest {
                    model: model.clone(),
                    session_id,
                    prompt_tokens: prompt_tokens.to_vec(),
                    block_table: block_table.clone(),
                    sampling_logits: sampling,
                },
                progress,
            )?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let mut mapped = |event| progress(metal_progress(event));
                metal.prefill_tokens_with_progress(
                    model, session_id, prompt_tokens, block_table, sampling, &mut mapped,
                )
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
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
