//! What a load would resolve, without changing the library.

use super::{Library, ModelLoadOptions};
use crate::{Model, ModelDescriptor, ModelMemoryEstimate, Result, RuntimeConfig};

/// The configuration and memory a load would use, computed without
/// reserving memory or replacing the library's engine.
#[derive(Clone, Debug)]
pub struct LoadPreview {
    /// The runtime configuration, with the K/V cache the load sizes.
    pub config: RuntimeConfig,
    /// The memory the load plans for.
    pub estimate: ModelMemoryEstimate,
    /// Bytes the resident copy of the model holds and a reload frees.
    pub released_bytes: u64,
}

impl Library {
    /// Resolves what loading `descriptor` with `options` would use without
    /// changing the library. `resident` is the same model when it is already
    /// loaded: its memory counts as free, since loading it again with other
    /// options replaces it, instead of being counted against it twice.
    pub fn preview_load(
        &self,
        descriptor: &ModelDescriptor,
        options: ModelLoadOptions,
        resident: Option<&Model>,
    ) -> Result<LoadPreview> {
        let probe = self.engine()?;
        let released_bytes = match resident {
            Some(model) => model.inner.memory.released_bytes()?,
            None => 0,
        };
        let mut memory = probe.memory_snapshot()?;
        memory.available_bytes =
            memory.available_bytes.map(|bytes| bytes.saturating_add(released_bytes));
        let committed = self.memory.committed_bytes()?.saturating_sub(released_bytes);
        let target = probe.target();
        let mut config = self.resolve_cache(descriptor, options, &probe, &memory, committed);
        // The cache the load is given: an automatic one may join a pool that
        // compatible models already share, and keeps that pool's size.
        let initial = descriptor.memory_estimate_for(&config, &target);
        config.kv_cache =
            self.caches
                .preview(&target, config.kv_cache, initial, options.kv_blocks.is_some());
        let estimate = descriptor.memory_estimate_for(&config, &target);
        Ok(LoadPreview { config, estimate, released_bytes })
    }
}
