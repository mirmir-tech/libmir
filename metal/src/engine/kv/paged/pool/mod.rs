use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::engine::{Array, Error, KvPageFormat, Result, Stream};

mod growth;
mod key;
pub(super) use growth::target as growth_target;
pub(super) use key::ArenaKey;

#[derive(Debug)]
pub(super) enum ArenaPages {
    Native {
        keys: Array,
        values: Array,
    },
    Quantized {
        keys: Array,
        values: Array,
        key_scales: Array,
        value_scales: Array,
    },
}

impl ArenaPages {
    pub(super) const fn keys(&self) -> &Array {
        match self {
            Self::Native { keys, .. } | Self::Quantized { keys, .. } => keys,
        }
    }

    pub(super) const fn values(&self) -> &Array {
        match self {
            Self::Native { values, .. } | Self::Quantized { values, .. } => values,
        }
    }

    pub(super) const fn scales(&self) -> Option<(&Array, &Array)> {
        match self {
            Self::Native { .. } => None,
            Self::Quantized { key_scales, value_scales, .. } => Some((key_scales, value_scales)),
        }
    }

    pub(super) fn arrays(&self) -> impl Iterator<Item = &Array> {
        let scales = match self {
            Self::Native { .. } => [None, None],
            Self::Quantized { key_scales, value_scales, .. } => {
                [Some(key_scales), Some(value_scales)]
            },
        };
        [self.keys(), self.values()].into_iter().chain(scales.into_iter().flatten())
    }
}

/// Per-layer K/V page arena indexed directly by runtime `BlockId`s, holding
/// pages `0..capacity` up to the highest one bound so far (see `growth`).
#[derive(Debug)]
pub(super) struct Arena {
    pub(super) pages: ArenaPages,
    pub(super) capacity: usize,
    pub(super) page_size: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    /// Test-only slot tracking for legacy multi-session tests that do not bind
    /// a `runtime::kv::BlockTable`. Never exists in production builds.
    #[cfg(test)]
    pub(super) unbound_refs: Vec<usize>,
}

/// Per-model registry of fixed-capacity layer arenas.
///
/// Strong references keep cached prefix blocks resident across sessions for the
/// lifetime of the loaded model; dropping the model drops its pool and releases
/// every arena buffer.
#[derive(Debug, Default)]
pub struct PagedArenaPool {
    arenas: Mutex<HashMap<ArenaKey, Arc<Mutex<Arena>>>>,
}

impl PagedArenaPool {
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
        let key = ArenaKey::new(
            layer,
            page_size,
            format,
            dimensions[1],
            dimensions[3],
            keys.native().dtype()?,
        )?;
        let mut arenas = self
            .arenas
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
        if let Some(arena) = arenas.get(&key) {
            return Ok(Arc::clone(arena));
        }
        let arena = Arc::new(Mutex::new(create(key, capacity, stream)?));
        arenas.insert(key, Arc::clone(&arena));
        drop(arenas);
        Ok(arena)
    }

    /// Whether at least one layer arena has already been allocated.
    pub(crate) fn is_populated(&self) -> Result<bool> {
        let arenas = self
            .arenas
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
        Ok(!arenas.is_empty())
    }

    pub(crate) fn detach_evaluated_graphs(&self, stream: &Stream) -> Result<()> {
        let arenas = {
            let arenas = self
                .arenas
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
            arenas.values().cloned().collect::<Vec<_>>()
        };
        for arena in arenas {
            let arena = arena
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena lock was poisoned".into()))?;
            for array in arena.pages.arrays() {
                array.detach_graph(stream)?;
            }
        }
        Ok(())
    }

    pub(crate) fn eval_with_graph_roots(&self, roots: &[&Array], stream: &Stream) -> Result<()> {
        let arenas = {
            let arenas = self
                .arenas
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
            arenas.values().cloned().collect::<Vec<_>>()
        };
        let mut paged = Vec::with_capacity(arenas.len() * 4);
        for arena in arenas {
            let arena = arena
                .lock()
                .map_err(|_| Error::InvalidModel("paged arena lock was poisoned".into()))?;
            for array in arena.pages.arrays() {
                paged.push(Array::from_native(array.native().clone())?);
            }
        }
        let mut all = Vec::with_capacity(roots.len() + paged.len());
        all.extend_from_slice(roots);
        all.extend(paged.iter());
        stream.eval_many(&all)
    }

    #[cfg(test)]
    pub(crate) fn resident_arenas(&self) -> Result<usize> {
        let arenas = self
            .arenas
            .lock()
            .map_err(|_| Error::InvalidModel("paged arena pool lock was poisoned".into()))?;
        Ok(arenas.len())
    }
}

fn create(key: ArenaKey, capacity: usize, stream: &Stream) -> Result<Arena> {
    if capacity == 0 {
        return Err(Error::KvPageCapacity {
            layer: key.layer,
            required: 1,
            maximum: 0,
        });
    }
    let shape = mirtal::Shape::new([key.kv_heads, capacity, key.page_size, key.stored_dim])?;
    let graph = stream.native().graph();
    let keys = Array::from_native(graph.full(&shape, 0.0, key.dtype)?)?;
    let values = Array::from_native(graph.full(&shape, 0.0, key.dtype)?)?;
    let pages = if key.format.quantized() {
        let scale_shape = mirtal::Shape::new([key.kv_heads, capacity, key.page_size])?;
        ArenaPages::Quantized {
            keys,
            values,
            key_scales: Array::from_native(graph.full(
                &scale_shape,
                1.0,
                mirtal::DType::Float32,
            )?)?,
            value_scales: Array::from_native(graph.full(
                &scale_shape,
                1.0,
                mirtal::DType::Float32,
            )?)?,
        }
    } else {
        ArenaPages::Native { keys, values }
    };
    Ok(Arena {
        pages,
        capacity,
        page_size: key.page_size,
        kv_heads: key.kv_heads,
        head_dim: key.head_dim,
        #[cfg(test)]
        unbound_refs: vec![0; capacity],
    })
}

#[cfg(test)]
pub(super) fn same(left: &Arc<Mutex<Arena>>, right: &Arc<Mutex<Arena>>) -> bool {
    Arc::ptr_eq(left, right)
}
