mod allocation;
mod write;

use std::sync::{Arc, Mutex, MutexGuard};

use crate::engine::{
    Array, Error, KvPageFormat, PagedKvContext, Result, Stream,
    attention::PagedAttentionScratch,
    kernels::{PreparedPageWrite, PreparedQuantizedPageWrite},
};

#[derive(Debug)]
pub(super) struct PagedStore {
    storage: Option<Storage>,
    page_size: usize,
    allocation_step: usize,
    reserve_pages: usize,
    attention_scratch: Arc<PagedAttentionScratch>,
    page_write: Option<Box<PreparedPageWrite>>,
    quantized_page_write: Option<Box<PreparedQuantizedPageWrite>>,
    format: KvPageFormat,
}

#[derive(Debug)]
struct Storage {
    arena: Arc<Mutex<Arena>>,
    table: Array,
    page_ids: Vec<u32>,
    table_capacity: usize,
    identity: bool,
}

#[derive(Debug)]
struct Arena {
    keys: Array,
    values: Array,
    key_scales: Option<Array>,
    value_scales: Option<Array>,
    capacity: usize,
    page_size: usize,
    kv_heads: usize,
    head_dim: usize,
    references: Vec<usize>,
}

impl PagedStore {
    pub(super) fn new(
        page_size: usize,
        step: usize,
        reserve_tokens: usize,
        format: KvPageFormat,
    ) -> Self {
        Self {
            storage: None,
            page_size,
            allocation_step: step.div_ceil(page_size).max(1),
            reserve_pages: reserve_tokens.div_ceil(page_size).max(1),
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
            quantized_page_write: None,
            format,
        }
    }

    pub(super) const fn active(&self) -> bool {
        self.storage.is_some()
    }

    pub(super) fn fragmented(&self) -> bool {
        self.storage.as_ref().is_some_and(|storage| !storage.identity)
    }

    pub(super) fn reserve(&mut self, tokens: usize) {
        self.reserve_pages = self.reserve_pages.max(tokens.div_ceil(self.page_size));
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        self.release()
    }

    pub(super) fn snapshot_at(&self, tokens: usize) -> Result<Self> {
        let storage = self
            .storage
            .as_ref()
            .map(|storage| -> Result<Storage> {
                let pages = tokens.div_ceil(self.page_size);
                if pages > storage.page_ids.len() {
                    return Err(Error::InvalidModel(
                        "paged snapshot exceeds initialized storage".into(),
                    ));
                }
                let page_ids = storage.page_ids[..pages].to_vec();
                let mut arena = lock(&storage.arena)?;
                for page in &page_ids {
                    arena.references[usize::try_from(*page)?] += 1;
                }
                drop(arena);
                Ok(Storage {
                    arena: Arc::clone(&storage.arena),
                    table: Array::from_native(storage.table.native().clone())?,
                    identity: page_ids
                        .iter()
                        .enumerate()
                        .all(|(index, page)| usize::try_from(*page) == Ok(index)),
                    page_ids,
                    table_capacity: storage.table_capacity,
                })
            })
            .transpose()?;
        Ok(Self {
            storage,
            page_size: self.page_size,
            allocation_step: self.allocation_step,
            reserve_pages: self.reserve_pages,
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
            quantized_page_write: None,
            format: self.format,
        })
    }

    #[cfg(test)]
    pub(super) fn page_count(&self) -> usize {
        self.storage.as_ref().map_or(0, |storage| storage.page_ids.len())
    }

    pub(super) fn update(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        allocation::ensure(self, keys, values, offset, stream)?;
        match self.format {
            KvPageFormat::Native => {
                let prepared =
                    self.page_write.get_or_insert_with(|| Box::new(PreparedPageWrite::default()));
                let storage = self.storage.as_mut().ok_or(Error::NullHandle("paged storage"))?;
                write::native(keys, values, offset, stream, storage, prepared)?;
            },
            KvPageFormat::Int8PerTokenHead => {
                let prepared = self
                    .quantized_page_write
                    .get_or_insert_with(|| Box::new(PreparedQuantizedPageWrite::default()));
                let storage = self.storage.as_mut().ok_or(Error::NullHandle("paged storage"))?;
                write::quantized(keys, values, offset, stream, storage, prepared)?;
            },
        }
        Ok(())
    }

    pub(super) fn context(&self, tokens: usize, stream: &Stream) -> Result<(Array, Array)> {
        if self.format.quantized() {
            return Err(Error::InvalidModel(
                "quantized K/V must be consumed by native paged attention".into(),
            ));
        }
        let storage = self.storage.as_ref().ok_or(Error::NullHandle("paged storage"))?;
        let arena = lock(&storage.arena)?;
        if tokens > storage.page_ids.len() * self.page_size {
            return Err(Error::InvalidModel("paged context exceeds initialized storage".into()));
        }
        let graph = stream.native().graph();
        let pages = tokens.div_ceil(self.page_size);
        let (keys, values, capacity) = if storage.identity {
            (arena.keys.native().clone(), arena.values.native().clone(), arena.capacity)
        } else {
            let ids = graph.slice(storage.table.native(), &[0], &[pages])?;
            (
                graph.take(arena.keys.native(), &ids, 1)?,
                graph.take(arena.values.native(), &ids, 1)?,
                pages,
            )
        };
        let shape =
            mirtal::Shape::new([1, arena.kv_heads, capacity * self.page_size, arena.head_dim])?;
        let keys = graph.reshape(&keys, &shape)?;
        let values = graph.reshape(&values, &shape)?;
        let stop = [1, arena.kv_heads, tokens, arena.head_dim];
        let keys = graph.slice(&keys, &[0, 0, 0, 0], &stop)?;
        let values = graph.slice(&values, &[0, 0, 0, 0], &stop)?;
        drop(arena);
        Ok((Array::from_native(keys)?, Array::from_native(values)?))
    }

    pub(super) fn context_for_attention(
        &self,
        tokens: usize,
        stream: &Stream,
    ) -> Result<PagedKvContext> {
        let storage = self.storage.as_ref().ok_or(Error::NullHandle("paged storage"))?;
        let arena = lock(&storage.arena)?;
        let offset = mirtal::Array::from_slice(&[u32::try_from(tokens)?], [1])?;
        let mut dependencies = vec![arena.keys.native(), arena.values.native()];
        dependencies.extend(arena.key_scales.as_ref().map(Array::native));
        dependencies.extend(arena.value_scales.as_ref().map(Array::native));
        let dependency = stream.native().graph().depends(&offset, &dependencies)?;
        let key_pages = arena.keys.native().clone();
        let value_pages = arena.values.native().clone();
        let key_scales = clone_optional(arena.key_scales.as_ref())?;
        let value_scales = clone_optional(arena.value_scales.as_ref())?;
        drop(arena);
        Ok(PagedKvContext {
            key_pages: Array::from_native(key_pages)?,
            value_pages: Array::from_native(value_pages)?,
            key_scales,
            value_scales,
            page_table: Array::from_native(storage.table.native().clone())?,
            page_dependency: Array::from_native(dependency)?,
            scratch: Arc::clone(&self.attention_scratch),
            page_size: self.page_size,
            context_tokens: tokens,
            fragmented: !storage.identity,
        })
    }

    pub(super) const fn quantized(&self) -> bool {
        self.format.quantized()
    }

    fn release(&mut self) -> Result<()> {
        if let Some(storage) = self.storage.take() {
            let mut arena = lock(&storage.arena)?;
            for page in storage.page_ids {
                let count = &mut arena.references[usize::try_from(page)?];
                *count = count.saturating_sub(1);
            }
            drop(arena);
        }
        Ok(())
    }
}

impl Drop for PagedStore {
    fn drop(&mut self) {
        drop(self.release());
    }
}

fn lock(arena: &Arc<Mutex<Arena>>) -> Result<MutexGuard<'_, Arena>> {
    arena
        .lock()
        .map_or_else(|_| Err(Error::InvalidModel("paged arena lock was poisoned".into())), Ok)
}

fn clone_optional(array: Option<&Array>) -> Result<Option<Array>> {
    array.map(|array| Array::from_native(array.native().clone())).transpose()
}
