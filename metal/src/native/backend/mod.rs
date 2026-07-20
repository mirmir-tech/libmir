mod batch;
mod execution;
mod load;
mod runtime;
mod vision;
mod worker;

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use ::runtime::{RuntimeError, backend::ModelHandle};

use self::worker::ModelClient;
use super::{
    error::{Error, Result},
    model::LoadedModel,
};
use crate::MetalConfig;

#[derive(Debug, Default)]
struct ModelRegistry {
    models: HashMap<String, ModelClient>,
}

#[derive(Debug, Clone)]
pub struct MetalBackend {
    models: Arc<Mutex<ModelRegistry>>,
    profile_decode: Arc<AtomicBool>,
    config: Arc<MetalConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct MetalMemoryStats {
    pub active: u64,
    pub cached: u64,
    pub limit: u64,
    pub recommended: Option<u64>,
}

impl MetalBackend {
    #[must_use]
    pub fn new(config: MetalConfig) -> Self {
        Self {
            models: Arc::default(),
            profile_decode: Arc::default(),
            config: Arc::new(config),
        }
    }

    /// Creates a backend after validating the requested global K/V storage
    /// format.
    pub fn try_new(config: MetalConfig) -> std::result::Result<Self, RuntimeError> {
        crate::engine::KvPageFormat::resolve(config.kv_cache.dtype)
            .map_err(|error| RuntimeError::Backend(error.to_string()))?;
        Ok(Self::new(config))
    }

    pub fn set_profile_decode(&self, enabled: bool) {
        self.profile_decode.store(enabled, Ordering::Relaxed);
    }

    pub fn clear_memory_cache(&self) -> std::result::Result<(), RuntimeError> {
        Ok(Self::clear_memory_cache_inner()?)
    }

    pub fn memory_stats(&self) -> std::result::Result<MetalMemoryStats, RuntimeError> {
        Ok(Self::memory_stats_inner()?)
    }

    pub fn with_generation_scope<T>(&self, run: impl FnOnce() -> T) -> T {
        run()
    }

    pub fn embed_tokens(
        &self,
        model: &ModelHandle,
        inputs: &[Vec<u32>],
        dimensions: usize,
    ) -> std::result::Result<Vec<Vec<f32>>, RuntimeError> {
        let lookup = model.id.clone();
        let inputs = inputs.to_vec();
        Ok(self.with_model(&lookup, move |loaded| {
            inputs.iter().map(|tokens| loaded.embed(tokens, dimensions)).collect()
        })?)
    }

    pub fn score_tokens(
        &self,
        model: &ModelHandle,
        inputs: &[Vec<u32>],
    ) -> std::result::Result<Vec<f32>, RuntimeError> {
        let lookup = model.id.clone();
        let inputs = inputs.to_vec();
        Ok(self.with_model(&lookup, move |loaded| {
            inputs.iter().map(|tokens| loaded.score(tokens)).collect()
        })?)
    }

    pub fn clear_prefix_cache(&self, model: &ModelHandle) -> std::result::Result<(), RuntimeError> {
        Ok(self.with_model(&model.id, move |loaded| {
            loaded.clear_prefix_cache();
            Ok(())
        })?)
    }

    pub fn release_session(
        &self,
        model: &ModelHandle,
        session: uuid::Uuid,
    ) -> std::result::Result<(), RuntimeError> {
        Ok(self.with_model(&model.id, move |loaded| {
            loaded.release_session(session);
            Ok(())
        })?)
    }

    pub fn unload_model(&self, model: &ModelHandle) -> std::result::Result<bool, RuntimeError> {
        Ok(self.unload_model_inner(model)?)
    }

    fn unload_model_inner(&self, model: &ModelHandle) -> Result<bool> {
        let client = self.models.lock()?.models.remove(&model.id);
        let Some(client) = client else {
            return Ok(false);
        };
        client.shutdown()?;
        Ok(true)
    }

    fn clear_memory_cache_inner() -> Result<()> {
        crate::engine::clear_memory_cache()?;
        Ok(())
    }

    fn memory_stats_inner() -> Result<MetalMemoryStats> {
        let memory = crate::engine::memory_stats()?;
        Ok(MetalMemoryStats {
            active: u64::try_from(memory.active)?,
            cached: u64::try_from(memory.cached)?,
            limit: u64::try_from(memory.limit)?,
            recommended: memory.recommended.map(u64::try_from).transpose()?,
        })
    }

    fn with_model<T>(
        &self,
        id: &str,
        run: impl FnOnce(&mut LoadedModel) -> Result<T> + Send + 'static,
    ) -> Result<T>
    where
        T: Send + 'static,
    {
        let model = self
            .models
            .lock()?
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| Error::ModelNotLoaded(id.to_owned()))?;
        model.run(run)
    }

    fn with_model_progress<T>(
        &self,
        id: &str,
        run: impl FnOnce(&mut LoadedModel, &mut dyn FnMut(crate::MetalProgressEvent)) -> Result<T>
        + Send
        + 'static,
        progress: &mut dyn FnMut(crate::MetalProgressEvent),
    ) -> Result<T>
    where
        T: Send + 'static,
    {
        let model = self
            .models
            .lock()?
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| Error::ModelNotLoaded(id.to_owned()))?;
        model.run_with_progress(run, progress)
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new(MetalConfig::default())
    }
}
