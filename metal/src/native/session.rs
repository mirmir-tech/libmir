use crate::engine::{Array, DecoderCache};

#[derive(Debug)]
pub(super) struct SessionState {
    pub(super) cache: DecoderCache,
    pub(super) position: usize,
    pub(super) rope_position_delta: i32,
    pub(super) pending: Option<PendingDecode>,
}

impl SessionState {
    pub(super) const fn new(cache: DecoderCache) -> Self {
        Self {
            cache,
            position: 0,
            rope_position_delta: 0,
            pending: None,
        }
    }

    pub(super) const fn from_prefix(cache: DecoderCache, position: usize) -> Self {
        Self {
            cache,
            position,
            rope_position_delta: 0,
            pending: None,
        }
    }

    pub(super) fn model_position(&self) -> crate::native::error::Result<usize> {
        let position = i64::try_from(self.position)? + i64::from(self.rope_position_delta);
        Ok(usize::try_from(position)?)
    }
}

#[derive(Debug)]
pub(super) struct PendingDecode {
    pub(super) token_id: u32,
    pub(super) logits: Array,
}
