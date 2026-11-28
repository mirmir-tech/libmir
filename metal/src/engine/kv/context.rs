use std::sync::Arc;

use super::{Array, PagedAttention};
use crate::engine::attention::PagedAttentionScratch;

#[derive(Debug)]
pub struct KvContext {
    pub keys: Array,
    pub values: Array,
    pub paged: Option<PagedKvContext>,
    pub mask: Option<Array>,
}

/// Optional per-token quantization scales attached to a paged K/V context.
#[derive(Debug)]
pub enum PagedScales {
    None,
    PerTokenHead { keys: Array, values: Array },
}

impl PagedScales {
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        matches!(self, Self::PerTokenHead { .. })
    }

    #[must_use]
    pub const fn pair(&self) -> Option<(&Array, &Array)> {
        match self {
            Self::None => None,
            Self::PerTokenHead { keys, values } => Some((keys, values)),
        }
    }
}

/// Physical arrangement of a session's pages inside an arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLayout {
    Contiguous,
    Fragmented,
}

impl PageLayout {
    #[must_use]
    pub const fn is_fragmented(self) -> bool {
        matches!(self, Self::Fragmented)
    }
}

#[derive(Debug)]
pub struct PagedKvContext {
    pub key_pages: Array,
    pub value_pages: Array,
    pub scales: PagedScales,
    pub page_table: Array,
    pub page_dependency: Array,
    pub(crate) scratch: Arc<PagedAttentionScratch>,
    pub page_size: usize,
    pub context_tokens: usize,
    pub(crate) layout: PageLayout,
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
        let (key_scales, value_scales) = match &self.scales {
            PagedScales::None => (None, None),
            PagedScales::PerTokenHead { keys, values } => (Some(keys), Some(values)),
        };
        PagedAttention {
            key_pages: &self.key_pages,
            value_pages: &self.value_pages,
            key_scales,
            value_scales,
            page_table: &self.page_table,
            page_dependency: &self.page_dependency,
            page_size: self.page_size,
            context_tokens: self.context_tokens,
        }
    }

    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        self.scales.is_quantized()
    }

    #[must_use]
    pub const fn is_fragmented(&self) -> bool {
        self.layout.is_fragmented()
    }

    pub(crate) fn scratch(&self) -> &PagedAttentionScratch {
        &self.scratch
    }
}
