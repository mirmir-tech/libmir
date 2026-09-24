use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use foundation::model::BackendTarget;
use models::generation::GenerationOverrides;
use runtime::kv::CacheConfig;

use super::{
    Library, Model, ModelDescriptor, ModelInner, automatic_cache,
    cache_cohort::CacheCohort,
    memory_admission::{AdmissionTerms, ModelMemoryManager},
    memory_policy,
};
use crate::{Engine, ProgressEvent, Result, RuntimeConfig, scheduler::ModelCoordinator};

#[derive(Debug)]
pub(super) struct LibraryState {
    pub(super) engine: Option<Engine>,
    pub(super) model_config: Option<RuntimeConfig>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Controls safety overrides applied while loading one model.
pub struct ModelLoadOptions {
    /// Continue when the atomic memory admission estimate exceeds the safe
    /// accelerator budget. Backend allocation may still fail.
    pub allow_memory_overcommit: bool,
}

impl Library {
    /// Creates a library whose backend is initialized lazily on first use.
    #[must_use]
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(LibraryState { engine: None, model_config: None })),
            memory: ModelMemoryManager::default(),
            caches: super::cache::KvCachePools::default(),
            registry: super::kv_sizing::ModelRegistry::default(),
            config,
        }
    }

    /// Inspects and loads a model directory, reporting weight-loading progress.
    pub fn load(
        &self,
        path: impl AsRef<Path>,
        overrides: GenerationOverrides,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<Model> {
        self.load_with_options(path, overrides, ModelLoadOptions::default(), progress)
    }

    /// Loads a model with explicit resource-admission options.
    pub fn load_with_options(
        &self,
        path: impl AsRef<Path>,
        overrides: GenerationOverrides,
        options: ModelLoadOptions,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<Model> {
        let descriptor = ModelDescriptor::inspect(path, overrides)?;
        let _load = self.memory.serialize_load()?;
        let (engine, mut config, kv_sizing) = self.model_runtime(&descriptor)?;
        let manifest = descriptor.manifest_for(engine.target())?;
        let target = engine.target();
        let initial_estimate = descriptor.memory_estimate_for(&config, &target);
        let cache = self.caches.acquire(&target, config.kv_cache, initial_estimate);
        config.kv_cache = cache.config;
        let estimate = descriptor.memory_estimate_for(&config, &target);
        let memory = engine.memory_snapshot()?;
        let reservation = self.memory.reserve(
            manifest.id.clone(),
            estimate,
            cache.shared_memory,
            &memory,
            config.memory,
            AdmissionTerms {
                reclaimable_bytes: self.reclaimable_kv_bytes()?,
                allow_overcommit: options.allow_memory_overcommit,
            },
        )?;
        let post_load_reserve = resolved_post_load_reserve(&target, &config, &memory, estimate);
        let handle = engine.load_model_with_progress_and_reservation(
            &manifest,
            post_load_reserve,
            resolved_metal_cache(&target, &config),
            kv_sizing.sequence_capacity_blocks(),
            progress,
        )?;
        progress(ProgressEvent::initialize_runtime(0, 1, "initializing model runtime"));
        let coordinator = match ModelCoordinator::new(
            engine.clone(),
            handle.clone(),
            config.scheduler.clone(),
            config.kv_cache,
        ) {
            Ok(coordinator) => coordinator,
            Err(error) => {
                cleanup_failed_load(&engine, &handle);
                return Err(error);
            },
        };
        if let Err(error) = reservation.mark_resident() {
            cleanup_failed_load(&engine, &handle);
            return Err(error);
        }
        let model = Model {
            inner: Arc::new(ModelInner {
                descriptor,
                engine,
                handle,
                cache: cache.cache,
                cache_cohort: CacheCohort::for_backend(
                    &target,
                    config.scheduler.decode_batch_wait_us,
                    config.scheduler.max_batch_tokens,
                    config.kv_cache.block_size,
                ),
                coordinator,
                kv_sizing: Mutex::new(kv_sizing),
                registry: self.registry.clone(),
                memory: reservation,
                config,
            }),
        };
        self.registry.register(&model.inner)?;
        progress(ProgressEvent::initialize_runtime(1, 1, "model runtime initialized"));
        Ok(model)
    }

    /// Returns the backend, cache, and scheduler configuration used by this
    /// library.
    #[must_use]
    pub const fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns the accelerator backend selected for this library.
    pub fn backend_target(&self) -> Result<BackendTarget> {
        self.engine().map(|engine| engine.target())
    }

    /// Resolves the effective runtime configuration for a model.
    pub fn model_config(&self, descriptor: &ModelDescriptor) -> Result<RuntimeConfig> {
        self.model_runtime(descriptor).map(|(_, config, _)| config)
    }

    /// Resizes idle models with measured K/V caches to the memory other
    /// models leave: they grow after an unload and shrink after a load.
    /// Returns how many caches changed.
    pub fn rebalance_kv_caches(&self) -> Result<usize> {
        let mut resized = 0;
        for model in self.registry.all()? {
            if model.rebalance_kv_cache()? {
                resized += 1;
            }
        }
        Ok(resized)
    }

    /// Bytes the loaded models' measured K/V caches would give back to a
    /// new model: what they hold above the minimum share each keeps.
    pub fn reclaimable_kv_bytes(&self) -> Result<u64> {
        let mut bytes = 0_u64;
        for model in self.registry.all()? {
            bytes = bytes.saturating_add(model.reclaimable_kv_bytes()?);
        }
        Ok(bytes)
    }

    pub(super) fn engine(&self) -> Result<Engine> {
        let mut state = self.lock_state()?;
        if let Some(engine) = state.engine.as_ref() {
            return Ok(engine.clone());
        }
        let config = state.model_config.as_ref().unwrap_or(&self.config);
        let initialized = Engine::from_config(config)?;
        state.engine = Some(initialized.clone());
        drop(state);
        Ok(initialized)
    }

    fn model_runtime(
        &self,
        descriptor: &ModelDescriptor,
    ) -> Result<(Engine, RuntimeConfig, super::kv_sizing::KvSizing)> {
        let mut state = self.lock_state()?;
        let engine_config = state.model_config.as_ref().unwrap_or(&self.config).clone();
        let probe = match state.engine.take() {
            Some(engine) => engine,
            None => Engine::from_config(&engine_config)?,
        };
        let target = probe.target();
        let estimate = descriptor.memory_estimate_for(&self.config, &target);
        let memory = probe.memory_snapshot()?;
        let committed = self.memory.committed_bytes()?;
        let mut config = automatic_cache::resolve(&self.config, estimate, &memory, committed);
        let resizable = probe.kv_cache_resizable(&descriptor.manifest_for(target.clone())?)?;
        let kv_sizing = super::kv_sizing::KvSizing::for_load(
            &self.config, &mut config, estimate, &target, resizable,
        );
        let resolved_estimate = descriptor.memory_estimate_for(&config, &target);
        let engine = if engine_cache_compatible(&target, config.kv_cache, engine_config.kv_cache) {
            probe
        } else {
            Engine::from_config(&config)?
        };
        tracing::info!(
            automatic = self.config.automatic_kv_cache,
            sizing = ?kv_sizing,
            blocks = config.kv_cache.block_count,
            block_size = config.kv_cache.block_size,
            capacity_tokens = u64::from(config.kv_cache.block_count)
                .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX)),
            kv_bytes_per_token = estimate.kv_bytes_per_token,
            kv_cache_bytes = resolved_estimate.kv_cache_bytes,
            platform_reserve_bytes = memory_policy::platform_reserve(config.memory, &memory),
            transient_reserve_bytes = memory_policy::transient_reserve(resolved_estimate, &memory),
            planned_residency_bytes = memory_policy::planned_residency(resolved_estimate, &memory),
            memory_source = memory.source,
            "resolved model KV cache"
        );
        state.model_config = Some(config.clone());
        state.engine = Some(engine.clone());
        drop(state);
        Ok((engine, config, kv_sizing))
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, LibraryState>> {
        let Ok(state) = self.state.lock() else {
            return Err(
                runtime::RuntimeError::Config("library runtime lock is poisoned".into()).into()
            );
        };
        Ok(state)
    }
}

fn resolved_post_load_reserve(
    target: &BackendTarget,
    config: &RuntimeConfig,
    memory: &crate::MemorySnapshot,
    estimate: crate::ModelMemoryEstimate,
) -> Option<usize> {
    (*target == BackendTarget::Metal).then(|| {
        let bytes = estimate
            .kv_cache_bytes
            .saturating_add(estimate.workspace_bytes)
            .saturating_add(memory_policy::platform_reserve(config.memory, memory));
        usize::try_from(bytes).unwrap_or(usize::MAX)
    })
}

fn resolved_metal_cache(target: &BackendTarget, config: &RuntimeConfig) -> Option<CacheConfig> {
    (*target == BackendTarget::Metal).then_some(config.kv_cache)
}

fn engine_cache_compatible(
    target: &BackendTarget,
    resolved: CacheConfig,
    current: CacheConfig,
) -> bool {
    resolved == current
        || (*target == BackendTarget::Metal
            && resolved.block_size == current.block_size
            && resolved.dtype == current.dtype)
}

fn cleanup_failed_load(engine: &Engine, handle: &runtime::backend::ModelHandle) {
    if let Err(error) = engine.unload_model(handle) {
        tracing::warn!(%error, model = handle.id, "failed to roll back partial model load");
    }
    if let Err(error) = engine.clear_memory_cache() {
        tracing::warn!(%error, model = handle.id, "failed to clear memory after load rollback");
    }
}

#[cfg(test)]
#[path = "library/tests.rs"]
mod tests;
