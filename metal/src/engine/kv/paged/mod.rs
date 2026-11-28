mod allocation;
mod context;
#[cfg(test)]
mod inspection;
mod pool;
mod rows;
mod tail;
mod validation;
mod write;

use std::sync::{Arc, Mutex, MutexGuard};

use pool::Arena;
pub use pool::PagedArenaPool;
pub(super) use rows::write_rows;
use runtime::kv::BlockId;

use crate::engine::{
    Array, Error, KvPageFormat, Result, Stream,
    attention::PagedAttentionScratch,
    kernels::{PreparedPageWrite, PreparedQuantizedPageWrite},
};

#[derive(Debug)]
pub(super) struct PagedStore {
    storage: Option<Storage>,
    page_size: usize,
    capacity: usize,
    /// Runtime-assigned physical `BlockId`s (`0..capacity`). `None` only when
    /// a standalone engine test runs without a `runtime::kv::BlockTable`, in
    /// which case sequential indices `0, 1, 2, ...` are used up to `capacity`.
    bound_blocks: Option<Vec<u32>>,
    attention_scratch: Arc<PagedAttentionScratch>,
    page_write: Option<Box<PreparedPageWrite>>,
    quantized_page_write: Option<Box<PreparedQuantizedPageWrite>>,
    format: KvPageFormat,
    pool: Arc<PagedArenaPool>,
    layer: usize,
    /// A prefix snapshot's copy of its partial last page (see `tail`).
    tail: Option<Arc<tail::PagedTail>>,
}

#[derive(Debug)]
struct Storage {
    input_dtype: mirtal::DType,
    arena: Arc<Mutex<Arena>>,
    table: Array,
    page_ids: Vec<u32>,
    table_capacity: usize,
    /// Whether `page_ids` form one consecutive physical run inside the arena.
    contiguous: bool,
}

impl PagedStore {
    pub(super) fn new(
        page_size: usize,
        capacity: usize,
        format: KvPageFormat,
        pool: Arc<PagedArenaPool>,
        layer: usize,
    ) -> Self {
        Self {
            storage: None,
            page_size,
            capacity,
            bound_blocks: None,
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
            quantized_page_write: None,
            format,
            pool,
            layer,
            tail: None,
        }
    }

    pub(super) const fn active(&self) -> bool {
        self.storage.is_some()
    }

    pub(super) fn fragmented(&self) -> bool {
        self.storage.as_ref().is_some_and(|storage| !storage.contiguous)
    }

    /// Whether `token_pos` falls within the bound block table (or arena
    /// capacity for standalone sequential caches).
    pub(super) fn can_write_token(&self, token_pos: usize) -> bool {
        let blocks = self.bound_blocks.as_ref().map_or(self.capacity, Vec::len);
        token_pos / self.page_size < blocks
    }

    /// Binds runtime-allocated `BlockId`s to this layer's paged store,
    /// validating that every block falls within `0..capacity`.
    pub(super) fn bind_blocks(&mut self, blocks: &[BlockId], stream: &Stream) -> Result<()> {
        let mut ids = Vec::with_capacity(blocks.len());
        for &block in blocks {
            if usize::try_from(block.0)? >= self.capacity {
                return Err(Error::KvBlockOutOfRange {
                    layer: self.layer,
                    block: block.0,
                    capacity: self.capacity,
                });
            }
            ids.push(block.0);
        }
        if let Some(storage) = self.storage.as_mut() {
            allocation::sync_page_table(storage, &ids, self.capacity, stream)?;
        }
        self.bound_blocks = Some(ids);
        Ok(())
    }

    pub(super) fn detach_evaluated_graph(&self, stream: &Stream) -> Result<()> {
        // Binding more blocks appends to the table lazily, and no graph root
        // reaches the table until attention reads it: detaching it before then
        // would leave decode an array without a producer.
        if let Some(storage) = &self.storage
            && storage.table.native().allocation()?.is_some()
        {
            storage.table.detach_graph(stream)?;
        }
        Ok(())
    }

    #[cfg_attr(not(test), allow(clippy::unnecessary_wraps))]
    pub(super) fn reset(&mut self) -> Result<()> {
        #[cfg(test)]
        self.release_unbound()?;
        self.storage = None;
        self.bound_blocks = None;
        self.tail = None;
        Ok(())
    }

    pub(super) fn snapshot_at(&self, tokens: usize) -> Result<Self> {
        let pages = tokens.div_ceil(self.page_size);
        let storage = self
            .storage
            .as_ref()
            .map(|storage| -> Result<Storage> {
                if pages > storage.page_ids.len() {
                    return Err(Error::InvalidModel(
                        "paged snapshot exceeds initialized storage".into(),
                    ));
                }
                let page_ids = storage.page_ids[..pages].to_vec();
                #[cfg(test)]
                if self.bound_blocks.is_none() {
                    let mut arena = lock(&storage.arena)?;
                    for &page in &page_ids {
                        arena.unbound_refs[usize::try_from(page)?] += 1;
                    }
                    drop(arena);
                }
                Ok(Storage {
                    input_dtype: storage.input_dtype,
                    arena: Arc::clone(&storage.arena),
                    table: Array::from_native(storage.table.native().clone())?,
                    contiguous: single_run(&page_ids),
                    page_ids,
                    table_capacity: storage.table_capacity,
                })
            })
            .transpose()?;
        let bound_blocks =
            self.bound_blocks.as_ref().map(|ids| ids[..pages.min(ids.len())].to_vec());
        Ok(Self {
            storage,
            page_size: self.page_size,
            capacity: self.capacity,
            bound_blocks,
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
            quantized_page_write: None,
            format: self.format,
            pool: Arc::clone(&self.pool),
            layer: self.layer,
            tail: self.tail_at(tokens),
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

    pub(super) const fn quantized(&self) -> bool {
        self.format.quantized()
    }

    #[cfg(test)]
    fn release_unbound(&mut self) -> Result<()> {
        if self.bound_blocks.is_none()
            && let Some(storage) = self.storage.take()
        {
            let mut arena = lock(&storage.arena)?;
            for page in storage.page_ids {
                let slot = &mut arena.unbound_refs[usize::try_from(page)?];
                *slot = slot.saturating_sub(1);
            }
            drop(arena);
        }
        Ok(())
    }
}

#[cfg(test)]
impl Drop for PagedStore {
    fn drop(&mut self) {
        drop(self.release_unbound());
    }
}

fn lock(arena: &Arc<Mutex<Arena>>) -> Result<MutexGuard<'_, Arena>> {
    arena
        .lock()
        .map_or_else(|_| Err(Error::InvalidModel("paged arena lock was poisoned".into())), Ok)
}

/// Whether `page_ids` name consecutive physical pages.
fn single_run(page_ids: &[u32]) -> bool {
    page_ids.windows(2).all(|pair| pair[0].checked_add(1) == Some(pair[1]))
}
