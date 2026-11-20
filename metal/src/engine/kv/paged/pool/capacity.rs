use super::{Arena, PagedArenaPool};
use crate::engine::{Error, KvPageFormat, Result};

impl PagedArenaPool {
    /// Free pages up to `maximum`, counting pages the arena has not grown to.
    pub(crate) fn available_pages(
        &self,
        maximum: usize,
        layer: usize,
        kv_heads: usize,
        head_dim: usize,
        format: KvPageFormat,
    ) -> Result<usize> {
        Ok(self
            .usage(layer, kv_heads, head_dim, format)?
            .map_or(maximum, |(_, used)| maximum.saturating_sub(used)))
    }

    /// Free pages inside the arena's current capacity: writing them allocates
    /// no memory, since arenas keep their high-water capacity.
    pub(crate) fn allocated_free_pages(
        &self,
        layer: usize,
        kv_heads: usize,
        head_dim: usize,
        format: KvPageFormat,
    ) -> Result<usize> {
        Ok(self
            .usage(layer, kv_heads, head_dim, format)?
            .map_or(0, |(capacity, used)| capacity.saturating_sub(used)))
    }

    /// The layer arena's capacity and pages in use, if it exists.
    fn usage(
        &self,
        layer: usize,
        kv_heads: usize,
        head_dim: usize,
        format: KvPageFormat,
    ) -> Result<Option<(usize, usize)>> {
        let arena = {
            let arenas = self
                .arenas
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
            arenas.iter().find_map(|(key, arena)| {
                (key.layer == layer
                    && key.kv_heads == kv_heads
                    && key.head_dim == head_dim
                    && key.format == format)
                    .then(|| arena.upgrade())
                    .flatten()
            })
        };
        let Some(arena) = arena else {
            return Ok(None);
        };
        let arena: std::sync::MutexGuard<'_, Arena> = arena
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena lock was poisoned".into()))?;
        let used = arena.references.iter().filter(|count| **count > 0).count();
        Ok(Some((arena.capacity, used)))
    }
}
