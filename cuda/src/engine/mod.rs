mod batch;
mod execution;
mod model;
mod runner;
mod runtime;
mod trace;
mod vision;

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

use ::runtime::{kv::CacheConfig, scheduler::SchedulerConfig};

use self::model::LoadedModel;
use crate::{CudaBackend, CudaConfig, Error, Result};

#[derive(Clone)]
pub struct CudaEngine {
    backend: CudaBackend,
    cache: CacheConfig,
    session_config: crate::CudaModelSessionConfig,
    scheduler: SchedulerConfig,
    models: Arc<Mutex<HashMap<String, Arc<LoadedModel>>>>,
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
        })
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
