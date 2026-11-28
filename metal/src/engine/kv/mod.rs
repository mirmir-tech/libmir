mod context;
mod contiguous;
mod format;
pub mod history;
#[cfg(test)]
mod inspection;
mod lifecycle;
mod paged;
mod policy;
mod rows;
mod share;
mod sliding;
mod snapshot;
mod validation;

use std::sync::Arc;

pub use context::{KvContext, PageLayout, PagedContextMode, PagedKvContext, PagedScales};
pub use format::KvPageFormat;
pub use paged::PagedArenaPool;
pub use policy::{
    NATIVE_PAGED_ATTENTION_MIN_CONTEXT, native_paged_attention_mode, paged_attention_enabled,
    paged_attention_min_context,
};
use runtime::kv::BlockId;

use super::{Array, Error, PagedAttention, Result, Stream};

#[derive(Debug)]
pub(super) struct ContiguousStore {
    pub(super) keys: Option<Array>,
    pub(super) values: Option<Array>,
    pub(super) capacity: usize,
    pub(super) step: usize,
}

#[derive(Debug)]
pub(super) struct SlidingStore {
    pub(super) keys: Option<Array>,
    pub(super) values: Option<Array>,
    pub(super) capacity: usize,
    pub(super) write_index: usize,
    pub(super) step: usize,
    pub(super) max_context: usize,
    /// Tokens of the latest sliding update: a single token right after a
    /// chunk is a prompt's last token and keeps the rewind margin.
    pub(super) last_update_tokens: usize,
}

#[derive(Debug)]
enum KvStorage {
    Contiguous(ContiguousStore),
    Sliding(SlidingStore),
    Paged(paged::PagedStore),
}

#[derive(Debug)]
pub struct KvCache {
    history: Option<history::Row>,
    storage: KvStorage,
    offset: usize,
}

impl KvCache {
    pub fn new(step: usize) -> Result<Self> {
        Self::new_with_window(step, None)
    }

    pub fn new_with_window(step: usize, max_context: Option<usize>) -> Result<Self> {
        if step == 0 || max_context == Some(0) {
            return Err(Error::InvalidModel("KV cache dimensions must be positive".into()));
        }
        let storage = max_context.map_or_else(
            || {
                KvStorage::Contiguous(ContiguousStore {
                    keys: None,
                    values: None,
                    capacity: 0,
                    step,
                })
            },
            |max_context| {
                KvStorage::Sliding(SlidingStore {
                    keys: None,
                    values: None,
                    capacity: 0,
                    write_index: 0,
                    step,
                    max_context,
                    last_update_tokens: 0,
                })
            },
        );
        Ok(Self { history: None, storage, offset: 0 })
    }

    pub fn new_paged(step: usize, page_size: usize) -> Result<Self> {
        Self::new_paged_with_format(step, page_size, KvPageFormat::Native)
    }

    pub(crate) fn new_paged_with_format(
        step: usize,
        page_size: usize,
        format: KvPageFormat,
    ) -> Result<Self> {
        Self::new_paged_with_pool(step, page_size, format, Arc::new(PagedArenaPool::default()), 0)
    }

    pub(crate) fn new_paged_with_pool(
        step: usize,
        page_size: usize,
        format: KvPageFormat,
        pool: Arc<PagedArenaPool>,
        layer: usize,
    ) -> Result<Self> {
        Self::new_paged_with_pool_capacity(step, page_size, format, 4096, pool, layer)
    }

    pub(crate) fn new_paged_with_pool_capacity(
        step: usize,
        page_size: usize,
        format: KvPageFormat,
        max_pages: usize,
        pool: Arc<PagedArenaPool>,
        layer: usize,
    ) -> Result<Self> {
        if step == 0 || page_size == 0 {
            return Err(Error::InvalidModel("KV cache dimensions must be positive".into()));
        }
        let pages = paged::PagedStore::new(page_size, max_pages, format, pool, layer);
        Ok(Self {
            history: None,
            storage: KvStorage::Paged(pages),
            offset: 0,
        })
    }

    pub const fn offset(&self) -> Result<usize> {
        Ok(self.offset)
    }

    pub fn bind_blocks(
        &mut self,
        blocks: &[BlockId],
        token_len: Option<usize>,
        stream: &Stream,
    ) -> Result<()> {
        if let KvStorage::Paged(pages) = &mut self.storage {
            pages.bind_blocks(blocks, stream)?;
            if let Some(tokens) = token_len {
                self.offset = tokens;
                self.history = None;
            }
        }
        Ok(())
    }

    pub(crate) fn can_write_token(&self, token_pos: usize) -> bool {
        match &self.storage {
            KvStorage::Paged(pages) => pages.can_write_token(token_pos),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => true,
        }
    }

    pub fn reserve(&mut self, _tokens: usize) -> Result<()> {
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn physical_page_count(&self) -> usize {
        match &self.storage {
            KvStorage::Paged(pages) => pages.page_count(),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => 0,
        }
    }

    /// Whether the active pages already form more than one physical run.
    pub(crate) fn fragmented_pages(&self) -> bool {
        match &self.storage {
            KvStorage::Paged(pages) => pages.active() && pages.fragmented(),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => false,
        }
    }

    pub fn update(&mut self, keys: &Array, values: &Array, stream: &Stream) -> Result<KvContext> {
        self.update_inner(keys, values, stream, 0, PagedContextMode::Both)
    }

    #[cfg(test)]
    pub(crate) fn update_with_page_min_context(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        page_min_context: usize,
    ) -> Result<KvContext> {
        self.update_inner(keys, values, stream, page_min_context, PagedContextMode::Both)
    }

    #[cfg(test)]
    pub(crate) fn update_for_attention(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        page_min_context: usize,
    ) -> Result<KvContext> {
        self.update_inner(keys, values, stream, page_min_context, PagedContextMode::View)
    }

    pub(crate) fn update_for_attention_mode(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        page_min_context: usize,
        mode: PagedContextMode,
    ) -> Result<KvContext> {
        self.update_inner(keys, values, stream, page_min_context, mode)
    }
}

fn clone_array(input: Option<&Array>) -> Result<Option<Array>> {
    input.map(|array| Array::from_native(array.native().clone())).transpose()
}
