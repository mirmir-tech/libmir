mod allocation;
mod context;
#[cfg(test)]
mod inspection;
mod planning;
mod pool;
pub use planning::DecodeCapacity;
mod reservation;
mod rows;
mod validation;
mod write;

use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};

use pool::Arena;
pub use pool::PagedArenaPool;
pub(super) use rows::write_rows;

use crate::engine::{
    Array, Error, KvPageFormat, Result, Stream,
    attention::PagedAttentionScratch,
    kernels::{PreparedPageWrite, PreparedQuantizedPageWrite},
};

#[derive(Debug)]
pub(super) struct PagedStore {
    storage: Option<Storage>,
    page_size: usize,
    allocation_step: usize,
    reserve_pages: usize,
    max_pages: usize,
    attention_scratch: Arc<PagedAttentionScratch>,
    page_write: Option<Box<PreparedPageWrite>>,
    quantized_page_write: Option<Box<PreparedQuantizedPageWrite>>,
    format: KvPageFormat,
    pool: Arc<PagedArenaPool>,
    layer: usize,
}

#[derive(Debug)]
struct Storage {
    input_dtype: mirtal::DType,
    arena: Arc<Mutex<Arena>>,
    table: Array,
    page_ids: Vec<u32>,
    reserved_page_ids: Vec<u32>,
    table_capacity: usize,
    /// The logical pages occupy one run of physical pages, so the context
    /// is a slice of the arena; the run may start at any page.
    contiguous: bool,
    /// Identifies the storage as a page writer; see [`Arena::writers`].
    id: u64,
}

impl PagedStore {
    pub(super) fn new(
        page_size: usize,
        step: usize,
        reserve_tokens: usize,
        max_pages: usize,
        format: KvPageFormat,
        pool: Arc<PagedArenaPool>,
        layer: usize,
    ) -> Self {
        Self {
            storage: None,
            page_size,
            allocation_step: step.div_ceil(page_size).max(1),
            reserve_pages: reserve_tokens.div_ceil(page_size).max(1),
            max_pages,
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
            quantized_page_write: None,
            format,
            pool,
            layer,
        }
    }

    pub(super) const fn active(&self) -> bool {
        self.storage.is_some()
    }

    pub(super) fn fragmented(&self) -> bool {
        self.storage.as_ref().is_some_and(|storage| !storage.contiguous)
    }

    pub(super) fn reserve(&mut self, tokens: usize) {
        self.reserve_pages = self.reserve_pages.max(tokens.div_ceil(self.page_size));
    }

    pub(super) fn plan_contiguous(&mut self, tokens: usize) {
        self.reserve_pages = tokens.div_ceil(self.page_size).max(1);
    }

    pub(super) fn detach_evaluated_graph(&self, stream: &Stream) -> Result<()> {
        if let Some(storage) = &self.storage {
            storage.table.detach_graph(stream)?;
        }
        Ok(())
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
                    input_dtype: storage.input_dtype,
                    arena: Arc::clone(&storage.arena),
                    table: Array::from_native(storage.table.native().clone())?,
                    contiguous: single_run(&page_ids),
                    id: next_storage_id(),
                    page_ids,
                    reserved_page_ids: Vec::new(),
                    table_capacity: storage.table_capacity,
                })
            })
            .transpose()?;
        Ok(Self {
            storage,
            page_size: self.page_size,
            allocation_step: self.allocation_step,
            reserve_pages: self.reserve_pages,
            max_pages: self.max_pages,
            attention_scratch: Arc::new(PagedAttentionScratch::default()),
            page_write: None,
            quantized_page_write: None,
            format: self.format,
            pool: Arc::clone(&self.pool),
            layer: self.layer,
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

    fn release(&mut self) -> Result<()> {
        if let Some(storage) = self.storage.take() {
            let mut arena = lock(&storage.arena)?;
            for page in storage.page_ids.into_iter().chain(storage.reserved_page_ids) {
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

/// Whether `page_ids` name consecutive physical pages.
fn single_run(page_ids: &[u32]) -> bool {
    page_ids.windows(2).all(|pair| pair[0].checked_add(1) == Some(pair[1]))
}

/// A fresh storage identity; zero is never issued, so it marks an unwritten
/// page.
fn next_storage_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}
