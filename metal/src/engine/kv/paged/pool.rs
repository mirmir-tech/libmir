use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

use crate::engine::{Array, Error, KvPageFormat, Result, Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ArenaKey {
    layer: usize,
    page_size: usize,
    kv_heads: usize,
    head_dim: usize,
    stored_dim: usize,
    dtype: mirtal::DType,
    format: KvPageFormat,
}

#[derive(Debug)]
pub(super) struct Arena {
    pub(super) keys: Array,
    pub(super) values: Array,
    pub(super) key_scales: Option<Array>,
    pub(super) value_scales: Option<Array>,
    pub(super) capacity: usize,
    pub(super) page_size: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) references: Vec<usize>,
}

impl Arena {
    pub(super) fn allocate(&mut self) -> Result<u32> {
        let index = self
            .references
            .iter()
            .position(|count| *count == 0)
            .ok_or_else(|| Error::InvalidModel("paged arena has no free page".into()))?;
        self.references[index] = 1;
        Ok(u32::try_from(index)?)
    }

    pub(super) fn allocate_contiguous(&mut self, count: usize) -> Result<Vec<u32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if let Some(start) = contiguous_free_start(&self.references, count) {
            self.references[start..start + count].fill(1);
            let pages = (start..start + count)
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            return Ok(pages);
        }
        (0..count).map(|_| self.allocate()).collect()
    }

    pub(super) fn has_contiguous_free(&self, count: usize) -> bool {
        contiguous_free_start(&self.references, count).is_some()
    }
}

fn contiguous_free_start(references: &[usize], count: usize) -> Option<usize> {
    references
        .windows(count)
        .position(|window| window.iter().all(|references| *references == 0))
}

#[derive(Debug, Default)]
pub struct PagedArenaPool {
    arenas: Mutex<HashMap<ArenaKey, Weak<Mutex<Arena>>>>,
}

impl PagedArenaPool {
    pub(crate) fn available_pages(
        &self,
        maximum: usize,
        layer: usize,
        kv_heads: usize,
        head_dim: usize,
        format: KvPageFormat,
    ) -> Result<usize> {
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
            return Ok(maximum);
        };
        let used = {
            let arena = arena
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena lock was poisoned".into()))?;
            arena.references.iter().filter(|count| **count > 0).count()
        };
        Ok(maximum.saturating_sub(used))
    }

    pub(super) fn acquire(
        &self,
        layer: usize,
        page_size: usize,
        format: KvPageFormat,
        keys: &Array,
        capacity: usize,
        stream: &Stream,
    ) -> Result<Arc<Mutex<Arena>>> {
        let dimensions = keys.native().shape()?.dimensions().to_vec();
        let stored_dim = format.packed_words(dimensions[3])?;
        let dtype = match format {
            KvPageFormat::Native => keys.native().dtype()?,
            KvPageFormat::Int8PerTokenHead => mirtal::DType::Uint32,
        };
        let key = ArenaKey {
            layer,
            page_size,
            kv_heads: dimensions[1],
            head_dim: dimensions[3],
            stored_dim,
            dtype,
            format,
        };
        let mut arenas = self
            .arenas
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
        if let Some(arena) = arenas.get(&key).and_then(Weak::upgrade) {
            return Ok(arena);
        }
        let arena = Arc::new(Mutex::new(create(key, capacity, stream)?));
        arenas.insert(key, Arc::downgrade(&arena));
        drop(arenas);
        Ok(arena)
    }

    pub(crate) fn detach_evaluated_graphs(&self) -> Result<()> {
        let arenas = {
            let mut arenas = self
                .arenas
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
            arenas.retain(|_, arena| arena.strong_count() > 0);
            arenas.values().filter_map(Weak::upgrade).collect::<Vec<_>>()
        };
        for arena in arenas {
            let arena = arena
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena lock was poisoned".into()))?;
            arena.keys.native().detach_graph()?;
            arena.values.native().detach_graph()?;
            for scales in [&arena.key_scales, &arena.value_scales].into_iter().flatten() {
                scales.native().detach_graph()?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn resident_arenas(&self) -> Result<usize> {
        let mut arenas = self
            .arenas
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
        arenas.retain(|_, arena| arena.strong_count() > 0);
        Ok(arenas.len())
    }
}

fn create(key: ArenaKey, capacity: usize, stream: &Stream) -> Result<Arena> {
    let shape = mirtal::Shape::new([key.kv_heads, capacity, key.page_size, key.stored_dim])?;
    let graph = stream.native().graph();
    let scale_shape = mirtal::Shape::new([key.kv_heads, capacity, key.page_size])?;
    let (key_scales, value_scales) = if key.format.quantized() {
        (
            Some(Array::from_native(graph.full(&scale_shape, 1.0, mirtal::DType::Float32)?)?),
            Some(Array::from_native(graph.full(&scale_shape, 1.0, mirtal::DType::Float32)?)?),
        )
    } else {
        (None, None)
    };
    Ok(Arena {
        keys: Array::from_native(graph.full(&shape, 0.0, key.dtype)?)?,
        values: Array::from_native(graph.full(&shape, 0.0, key.dtype)?)?,
        key_scales,
        value_scales,
        capacity,
        page_size: key.page_size,
        kv_heads: key.kv_heads,
        head_dim: key.head_dim,
        references: vec![0; capacity],
    })
}

#[cfg(test)]
pub(super) fn same(left: &Arc<Mutex<Arena>>, right: &Arc<Mutex<Arena>>) -> bool {
    Arc::ptr_eq(left, right)
}

#[cfg(test)]
mod tests {
    use super::contiguous_free_start;

    #[test]
    fn prefers_a_run_large_enough_for_the_whole_tail() {
        assert_eq!(contiguous_free_start(&[1, 0, 0, 1, 0, 0, 0, 0], 3), Some(4));
        assert_eq!(contiguous_free_start(&[1, 0, 1, 0], 2), None);
    }
}
