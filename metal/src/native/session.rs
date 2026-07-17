use crate::engine::{Array, DecoderCache};

#[derive(Debug)]
pub(super) struct SessionState {
    pub(super) cache: DecoderCache,
    pub(super) position: usize,
    pub(super) pending: Option<PendingDecode>,
}

impl SessionState {
    pub(super) const fn new(cache: DecoderCache) -> Self {
        Self { cache, position: 0, pending: None }
    }

    pub(super) const fn from_prefix(cache: DecoderCache, position: usize) -> Self {
        Self { cache, position, pending: None }
    }
}

#[derive(Debug)]
pub(super) struct PendingDecode {
    pub(super) token_id: u32,
    pub(super) logits: Array,
}
