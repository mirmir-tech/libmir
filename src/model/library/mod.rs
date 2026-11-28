mod preview;
pub use preview::LoadPreview;
mod resolve;

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use foundation::model::BackendTarget;
use models::generation::GenerationOverrides;

use self::resolve::{cleanup_failed_load, resolved_metal_cache, resolved_post_load_reserve};
use super::{
    Library, Model, ModelDescriptor, ModelInner, ModelSource,
    cache::cohort::CacheCohort,
    memory::admission::{AdmissionTerms, ModelMemoryManager},
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
    /// A fixed K/V block count for this model in place of the library's
    /// sizing, for models whose requests need far less cache than others,
    /// such as rerankers.
    pub kv_blocks: Option<std::num::NonZeroU32>,
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
        self.load_admitted(descriptor, options, progress)
    }

    /// Inspects and loads a [`ModelSource`] (directory or direct GGUF +
    /// optional `mmproj` files) with explicit resource-admission options.
    pub fn load_source(
        &self,
        source: ModelSource<'_>,
        overrides: GenerationOverrides,
        options: ModelLoadOptions,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<Model> {
        let descriptor = ModelDescriptor::inspect_source(source, overrides)?;
        self.load_admitted(descriptor, options, progress)
    }

    /// Admits and loads an already-inspected [`ModelDescriptor`] without
    /// re-reading checkpoint layout files.
    pub fn load_admitted(
        &self,
        descriptor: ModelDescriptor,
        options: ModelLoadOptions,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<Model> {
        let _load = self.memory.serialize_load()?;
        let (engine, mut config, kv_sizing, admitted) = self.model_runtime(&descriptor, options)?;
        let manifest = descriptor.manifest_for(engine.target())?;
        let target = engine.target();
        let initial_estimate = descriptor.memory_estimate_for(&config, &target);
        let cache = self.caches.acquire(
            target,
            config.kv_cache,
            initial_estimate,
            options.kv_blocks.is_some(),
        );
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
        let post_load_reserve = resolved_post_load_reserve(target, &config, &memory, estimate);
        let handle = engine.load_admitted(
            &manifest,
            &descriptor.inspected,
            admitted,
            post_load_reserve,
            resolved_metal_cache(target, &config),
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
                    target,
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
        self.model_config_with_options(descriptor, ModelLoadOptions::default())
    }

    /// Resolves the runtime configuration a load with `options` would use,
    /// without changing the library.
    pub fn model_config_with_options(
        &self,
        descriptor: &ModelDescriptor,
        options: ModelLoadOptions,
    ) -> Result<RuntimeConfig> {
        self.preview_load(descriptor, options, None).map(|preview| preview.config)
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

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, LibraryState>> {
        let Ok(state) = self.state.lock() else {
            return Err(
                runtime::RuntimeError::Config("library runtime lock is poisoned".into()).into()
            );
        };
        Ok(state)
    }
}

#[cfg(test)]
mod tests;
