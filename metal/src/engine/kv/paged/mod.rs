mod allocation;

use std::sync::{Arc, Mutex, MutexGuard};

use crate::engine::{
    Array, Error, PagedKvContext, Result, Stream,
    attention::PagedAttentionScratch,
    kernels::{PageWriteOptions, PreparedPageWrite},
};

#[derive(Debug)]
pub(super) struct PagedStore {
    storage: Option<Storage>,
    page_size: usize,
    allocation_step: usize,
    reserve_pages: usize,
    attention_scratch: Arc<PagedAttentionScratch>,
    page_write: Option<Box<PreparedPageWrite>>,
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
    capacity: usize,
    page_size: usize,
    kv_heads: usize,
    head_dim: usize,
    references: Vec<usize>,
}

impl PagedStore {
    pub(super) fn new(page_size: usize, step: usize, reserve_tokens: usize) -> Self {
        Self {
            storage: None,
            page_size,
            allocation_step: step.div_ceil(page_size).max(1),
            reserve_pages: reserve_tokens.div_ceil(page_size).max(1),
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
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

    pub(super) fn snapshot(&self) -> Result<Self> {
        let storage = self
            .storage
            .as_ref()
            .map(|storage| -> Result<Storage> {
                let mut arena = lock(&storage.arena)?;
                for page in &storage.page_ids {
                    arena.references[usize::try_from(*page)?] += 1;
                }
                drop(arena);
                Ok(Storage {
                    arena: Arc::clone(&storage.arena),
                    table: Array::from_native(storage.table.native().clone())?,
                    page_ids: storage.page_ids.clone(),
                    table_capacity: storage.table_capacity,
                    identity: storage.identity,
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
        })
    }

    pub(super) fn update(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        allocation::ensure(self, keys, values, offset, stream)?;
        let prepared =
            self.page_write.get_or_insert_with(|| Box::new(PreparedPageWrite::default()));
        let storage = self.storage.as_mut().ok_or(Error::NullHandle("paged storage"))?;
        let mut arena = lock(&storage.arena)?;
        let [page_keys, page_values] = stream.page_write(
            [
                keys.native(),
                values.native(),
                arena.keys.native(),
                arena.values.native(),
                storage.table.native(),
            ],
            PageWriteOptions {
                sequence: keys.native().shape()?.dimensions()[2],
                offset,
                kv_heads: arena.kv_heads,
                page_capacity: arena.capacity,
                page_size: arena.page_size,
                head_dim: arena.head_dim,
            },
            prepared,
        )?;
        arena.keys = Array::from_native(page_keys)?;
        arena.values = Array::from_native(page_values)?;
        drop(arena);
        Ok(())
    }

    pub(super) fn context(&self, tokens: usize, stream: &Stream) -> Result<(Array, Array)> {
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
        let dependency = stream
            .native()
            .graph()
            .depends(&offset, &[arena.keys.native(), arena.values.native()])?;
        let key_pages = arena.keys.native().clone();
        let value_pages = arena.values.native().clone();
        drop(arena);
        Ok(PagedKvContext {
            key_pages: Array::from_native(key_pages)?,
            value_pages: Array::from_native(value_pages)?,
            page_table: Array::from_native(storage.table.native().clone())?,
            page_dependency: Array::from_native(dependency)?,
            scratch: Arc::clone(&self.attention_scratch),
            page_size: self.page_size,
            context_tokens: tokens,
        })
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
