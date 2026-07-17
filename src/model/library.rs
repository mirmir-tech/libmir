use std::{path::Path, sync::Arc};

use models::generation::GenerationOverrides;
use runtime::kv::KvCache;

use super::{Library, Model, ModelDescriptor, ModelInner};
use crate::{Engine, ProgressEvent, Result, scheduler::DecodeCoordinator};

impl Library {
    /// Creates a library whose backend is initialized lazily on first use.
    #[must_use]
    pub fn new(config: crate::RuntimeConfig) -> Self {
        Self {
            engine: Arc::new(std::sync::Mutex::new(None)),
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
        let descriptor = ModelDescriptor::inspect(path, overrides)?;
        let engine = self.engine()?;
        let handle = engine
            .load_model_with_progress(&descriptor.manifest_for(engine.target())?, progress)?;
        let coordinator =
            DecodeCoordinator::new(engine.clone(), handle.clone(), self.config.scheduler.clone());
        Ok(Model {
            inner: Arc::new(ModelInner {
                descriptor,
                engine,
                handle,
                cache: std::sync::Mutex::new(KvCache::with_config(self.config.kv_cache)),
                coordinator,
                config: self.config.clone(),
            }),
        })
    }

    /// Returns the backend, cache, and scheduler configuration used by this
    /// library.
    #[must_use]
    pub const fn config(&self) -> &crate::RuntimeConfig {
        &self.config
    }

    pub(super) fn engine(&self) -> Result<Engine> {
        let Ok(mut engine) = self.engine.lock() else {
            return Err(
                runtime::RuntimeError::Config("library engine lock is poisoned".into()).into()
            );
        };
        if let Some(engine) = engine.as_ref() {
            return Ok(engine.clone());
        }
        let initialized = Engine::from_config(&self.config)?;
        *engine = Some(initialized.clone());
        drop(engine);
        Ok(initialized)
    }
}
