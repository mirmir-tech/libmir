use super::*;

/// Host metadata only: buffer extents are not allocator bytes or DRAM traffic.
#[derive(Debug, serde::Serialize)]
pub struct ArenaMetrics {
    pub layer: usize,
    pub capacity_pages: usize,
    pub owned_pages: usize,
    pub shared_pages: usize,
    pub buffer_bytes: usize,
}

impl PagedArenaPool {
    pub(crate) fn resident_metrics(&self) -> Result<Vec<ArenaMetrics>> {
        let arenas = self
            .arenas
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
        let mut metrics = Vec::new();
        for (key, weak) in arenas.iter() {
            let Some(arena) = weak.upgrade() else {
                continue;
            };
            let arena = super::super::lock(&arena)?;
            let buffer_bytes = [&arena.keys, &arena.values]
                .into_iter()
                .chain([&arena.key_scales, &arena.value_scales].into_iter().flatten())
                .try_fold(0_usize, |sum, array| {
                    sum.checked_add(array.byte_len()?).ok_or(Error::ShapeOverflow)
                })?;
            metrics.push(ArenaMetrics {
                layer: key.layer,
                capacity_pages: arena.capacity,
                owned_pages: arena.references.iter().filter(|&&n| n > 0).count(),
                shared_pages: arena.references.iter().filter(|&&n| n > 1).count(),
                buffer_bytes,
            });
        }
        drop(arenas);
        metrics.sort_unstable_by_key(|m| m.layer);
        Ok(metrics)
    }
}
