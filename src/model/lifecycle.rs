use std::sync::Arc;

use super::{Library, Model};
use crate::{Error, MemorySnapshot, Result};

impl Library {
    /// Returns current memory information from the lazily initialized backend.
    pub fn memory_snapshot(&self) -> Result<MemorySnapshot> {
        Ok(self.engine()?.memory_snapshot()?)
    }
}

impl Model {
    #[must_use]
    /// Returns whether model clones or sessions currently prevent unloading.
    pub fn is_in_use(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }

    /// Unloads backend resources when this is the model's sole remaining owner.
    pub fn unload(self) -> Result<()> {
        let Self { inner } = self;
        let inner = Arc::try_unwrap(inner).map_err(|_| Error::ModelInUse)?;
        let engine = inner.engine.clone();
        let handle = inner.handle.clone();
        engine.unload_model(&handle)?;
        drop(inner);
        engine.clear_memory_cache()?;
        Ok(())
    }
}
