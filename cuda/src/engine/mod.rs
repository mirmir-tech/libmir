mod batch;
#[allow(clippy::self_named_module_files)]
mod execution;
pub mod lowering;
mod model;
mod profile;
mod runner;
mod runtime;
mod trace;
mod vision;

use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use ::runtime::{kv::CacheConfig, scheduler::SchedulerConfig};
pub use execution::{CudaGenerationStepOutput, CudaPrefillBatch};

use self::model::LoadedModel;
use crate::{CudaBackend, CudaConfig, Error, Result, backend::ProfilerCapture};

#[derive(Clone)]
pub struct CudaEngine {
    backend: CudaBackend,
    cache: CacheConfig,
    session_config: crate::CudaModelSessionConfig,
    scheduler: SchedulerConfig,
    models: Arc<Mutex<HashMap<String, Arc<LoadedModel>>>>,
    profile_decode: Arc<AtomicBool>,
    profiler_capture: Arc<Mutex<Option<ProfilerCapture>>>,
}

#[derive(Debug, Clone)]
pub struct CudaMemoryStats {
    pub total: u64,
    pub available: u64,
    pub reserved: u64,
    pub used: u64,
    pub device: String,
    pub integrated: bool,
}

impl CudaEngine {
    pub fn new(config: CudaConfig, cache: CacheConfig) -> Result<Self> {
        Self::new_with_scheduler(config, cache, SchedulerConfig::default())
    }

    pub fn new_with_scheduler(
        config: CudaConfig,
        cache: CacheConfig,
        scheduler: SchedulerConfig,
    ) -> Result<Self> {
        let session_config = config.model_session;
        Ok(Self {
            backend: CudaBackend::new(config)?,
            cache,
            session_config,
            scheduler,
            models: Arc::new(Mutex::new(HashMap::new())),
            profile_decode: Arc::default(),
            profiler_capture: Arc::default(),
        })
    }

    pub fn set_profile_decode(&self, enabled: bool) -> Result<()> {
        if enabled {
            self.start_profiler_capture()?;
            self.profile_decode.store(true, Ordering::Relaxed);
            return Ok(());
        }
        self.profile_decode.store(false, Ordering::Relaxed);
        self.stop_profiler_capture()
    }

    pub fn start_profiler_capture(&self) -> Result<()> {
        let Ok(mut capture) = self.profiler_capture.lock() else {
            return Err(Error::State("CUDA profiler capture lock is poisoned".into()));
        };
        if capture.is_none() {
            *capture = Some(self.backend.start_profiler_capture()?);
        }
        Ok(())
    }

    pub fn stop_profiler_capture(&self) -> Result<()> {
        let Ok(mut capture) = self.profiler_capture.lock() else {
            return Err(Error::State("CUDA profiler capture lock is poisoned".into()));
        };
        let active = capture.take();
        drop(capture);
        active.map_or(Ok(()), ProfilerCapture::stop)
    }

    fn profile_decode(&self) -> bool {
        self.profile_decode.load(Ordering::Relaxed)
    }

    fn model(&self, id: &str) -> Result<Arc<LoadedModel>> {
        self.models()?
            .get(id)
            .cloned()
            .ok_or_else(|| Error::State(format!("model is not loaded: {id}")))
    }

    fn models(&self) -> Result<MutexGuard<'_, HashMap<String, Arc<LoadedModel>>>> {
        let Ok(models) = self.models.lock() else {
            return Err(Error::State("model registry lock is poisoned".into()));
        };
        Ok(models)
    }

    pub fn clear_model_sessions(&self, model_id: &str) -> Result<()> {
        self.model(model_id)?.clear_sessions()
    }

    #[must_use]
    /// Returns the prompt tokens processed by one CUDA prefill graph.
    pub fn prefill_chunk_tokens(&self) -> usize {
        self.session_config.prefill_chunk_tokens
    }

    pub fn release_session(&self, model_id: &str, session: uuid::Uuid) -> Result<()> {
        self.model(model_id)?.release_session(session)
    }

    pub fn unload_model(&self, model_id: &str) -> Result<bool> {
        let removed = self.models()?.remove(model_id);
        let unloaded = removed.is_some();
        drop(removed);
        Ok(unloaded)
    }

    pub fn clear_memory_cache(&self) -> Result<()> {
        self.backend.trim_memory_pool(0)
    }

    pub fn finish_startup_tuning(&self) {
        self.backend.finish_startup_tuning();
    }

    pub fn memory_stats(&self) -> Result<CudaMemoryStats> {
        let pool = self.backend.memory_pool_stats()?;
        let (available, total) = self.backend.memory_info()?;
        let device = self.backend.device_info();
        Ok(CudaMemoryStats {
            total: u64::try_from(total)?,
            available: u64::try_from(available)?,
            reserved: pool.reserved,
            used: pool.used,
            device: device.name.clone(),
            integrated: device.integrated,
        })
    }

    /// Returns immutable properties of the CUDA device selected by this engine.
    #[must_use]
    pub fn device_info(&self) -> &mircuda::DeviceInfo {
        self.backend.device_info()
    }

    pub fn embed_tokens(
        &self,
        model: &::runtime::backend::ModelHandle,
        inputs: &[Vec<u32>],
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let loaded = self.model(&model.id)?;
        if !matches!(loaded.task_plan, models::execution::TaskExecutionPlan::Embedding { .. }) {
            return Err(Error::State("loaded CUDA task is not an embedding model".into()));
        }
        let mut runner = loaded.prefill_runner()?;
        let model::ModelExecution::Embedding(task) = &mut runner.execution else {
            return Err(Error::State("loaded CUDA task does not expose embeddings".into()));
        };
        let result = inputs.iter().map(|tokens| task.embed(tokens, dimensions)).collect();
        drop(runner);
        result
    }

    pub fn score_tokens(
        &self,
        model: &::runtime::backend::ModelHandle,
        inputs: &[Vec<u32>],
    ) -> Result<Vec<f32>> {
        let loaded = self.model(&model.id)?;
        if !matches!(loaded.task_plan, models::execution::TaskExecutionPlan::SequenceScoring { .. })
        {
            return Err(Error::State("loaded CUDA task is not a sequence-scoring model".into()));
        }
        let mut runner = loaded.prefill_runner()?;
        let model::ModelExecution::SequenceScoring(task) = &mut runner.execution else {
            return Err(Error::State("loaded CUDA task does not expose sequence scores".into()));
        };
        let result = inputs.iter().map(|tokens| task.score(tokens)).collect();
        drop(runner);
        result
    }
}

impl fmt::Debug for CudaEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaEngine")
            .field("device", &self.backend.device_info().name)
            .field("cache", &self.cache)
            .field("session_config", &self.session_config)
            .field("scheduler", &self.scheduler)
            .finish_non_exhaustive()
    }
}
