mod contiguous;
mod paged;
mod policy;
mod sliding;

use std::sync::Arc;

pub use policy::{
    NATIVE_PAGED_ATTENTION_MIN_CONTEXT, native_paged_attention_mode, paged_attention_enabled,
    paged_attention_min_context,
};

use super::{Array, Error, PagedAttention, Result, Stream};
use crate::engine::attention::PagedAttentionScratch;

#[derive(Debug)]
pub struct KvContext {
    pub keys: Array,
    pub values: Array,
    pub paged: Option<PagedKvContext>,
    pub mask: Option<Array>,
}

#[derive(Debug)]
pub struct PagedKvContext {
    pub key_pages: Array,
    pub value_pages: Array,
    pub page_table: Array,
    pub page_dependency: Array,
    pub(crate) scratch: Arc<PagedAttentionScratch>,
    pub page_size: usize,
    pub context_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedContextMode {
    View,
    NativeIfFragmented,
    Native,
    Both,
}

impl PagedKvContext {
    #[must_use]
    pub fn attention(&self) -> PagedAttention<'_> {
        PagedAttention {
            key_pages: &self.key_pages,
            value_pages: &self.value_pages,
            page_table: &self.page_table,
            page_dependency: &self.page_dependency,
            page_size: self.page_size,
            context_tokens: self.context_tokens,
        }
    }

    pub(crate) fn scratch(&self) -> &PagedAttentionScratch {
        &self.scratch
    }
}

#[derive(Debug)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    pages: Option<paged::PagedStore>,
    offset: usize,
    capacity: usize,
    write_index: usize,
    step: usize,
    max_context: Option<usize>,
    reserve_tokens: usize,
}

impl KvCache {
    pub fn new(step: usize) -> Result<Self> {
        Self::new_with_window(step, None)
    }

    pub fn new_with_window(step: usize, max_context: Option<usize>) -> Result<Self> {
        Self::new_with_options(step, max_context, None)
    }

    pub fn new_paged(step: usize, page_size: usize) -> Result<Self> {
        Self::new_with_options(step, None, Some(page_size))
    }

    fn new_with_options(
        step: usize,
        max_context: Option<usize>,
        page_size: Option<usize>,
    ) -> Result<Self> {
        if step == 0 || max_context == Some(0) || page_size == Some(0) {
            return Err(Error::InvalidModel("KV cache dimensions must be positive".into()));
        }
        let reserve_tokens = usize::from(page_size.is_some()) * step;
        Ok(Self {
            keys: None,
            values: None,
            pages: page_size.map(|size| paged::PagedStore::new(size, step, reserve_tokens)),
            offset: 0,
            capacity: 0,
            write_index: 0,
            step,
            max_context,
            reserve_tokens,
        })
    }

    pub const fn offset(&self) -> Result<usize> {
        Ok(self.offset)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.keys = None;
        self.values = None;
        if let Some(pages) = self.pages.as_mut() {
            pages.reset()?;
        }
        self.offset = 0;
        self.capacity = 0;
        self.write_index = 0;
        Ok(())
    }

    pub fn reserve(&mut self, tokens: usize) -> Result<()> {
        self.reserve_tokens = self.reserve_tokens.max(tokens);
        if let Some(pages) = self.pages.as_mut() {
            pages.reserve(self.reserve_tokens);
        }
        Ok(())
    }

    pub fn snapshot_at(&self, offset: usize) -> Result<Self> {
        if offset > self.offset {
            return Err(Error::InvalidModel(
                "KV cache snapshot offset exceeds cached tokens".into(),
            ));
        }
        Ok(Self {
            keys: clone_array(self.keys.as_ref())?,
            values: clone_array(self.values.as_ref())?,
            pages: self.pages.as_ref().map(paged::PagedStore::snapshot).transpose()?,
            offset,
            capacity: self.capacity,
            write_index: self.write_index,
            step: self.step,
            max_context: self.max_context,
            reserve_tokens: self.reserve_tokens,
        })
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
