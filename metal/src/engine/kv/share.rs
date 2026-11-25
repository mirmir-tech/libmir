//! Contexts read by more than one layer: Gemma 4's E models attend later
//! layers over an earlier layer's keys and values in the same step.

use std::sync::Arc;

use super::{KvContext, PagedKvContext};
use crate::engine::{Array, Result};

impl KvContext {
    /// Another owner of the same context, for layers that attend over the
    /// keys and values of another layer (Gemma 4's E models).
    pub fn share(&self) -> Result<Self> {
        Ok(Self {
            keys: self.keys.snapshot()?,
            values: self.values.snapshot()?,
            paged: self.paged.as_ref().map(PagedKvContext::share).transpose()?,
            mask: self.mask.as_ref().map(Array::snapshot).transpose()?,
        })
    }
}

impl PagedKvContext {
    fn share(&self) -> Result<Self> {
        let optional = |array: Option<&Array>| array.map(Array::snapshot).transpose();
        Ok(Self {
            key_pages: self.key_pages.snapshot()?,
            value_pages: self.value_pages.snapshot()?,
            key_scales: optional(self.key_scales.as_ref())?,
            value_scales: optional(self.value_scales.as_ref())?,
            page_table: self.page_table.snapshot()?,
            page_dependency: self.page_dependency.snapshot()?,
            // Only a later layer of the same step reads it, after this
            // layer's attention, which its queries depend on.
            scratch: Arc::clone(&self.scratch),
            page_size: self.page_size,
            context_tokens: self.context_tokens,
            fragmented: self.fragmented,
        })
    }
}
