//! Prefix snapshots of one layer's K/V, at its current offset or rewound.

use super::{KvCache, clone_array};
use crate::engine::{Error, Result};

impl KvCache {
    /// The smallest offset [`Self::snapshot_at`] restores exactly. Pages keep
    /// every token. A sliding cache after a prefill chunk holds its window
    /// plus `SLIDING_REWIND_TOKENS` and the chunk chronologically, so it
    /// rewinds by that margin: enough for a request that shares the previous
    /// prompt but not its last template tokens (a chat turn's generation
    /// prompt, llama-benchy's context probe). A wrapped decode ring and an
    /// unpaged full cache only snapshot their current offset.
    pub(crate) fn prefix_floor(&self) -> usize {
        if self.pages.is_some() {
            return 0;
        }
        match (self.max_context, self.keys.as_ref()) {
            (Some(window), Some(_)) => {
                let stored = self.sliding_valid(window);
                if self.write_index < stored {
                    self.offset
                } else if stored >= self.offset {
                    0
                } else {
                    (self.offset - stored).saturating_add(window - 1).min(self.offset)
                }
            },
            _ => self.offset,
        }
    }

    pub fn snapshot_at(&self, offset: usize) -> Result<Self> {
        if offset > self.offset {
            return Err(Error::InvalidModel(
                "KV cache snapshot offset exceeds cached tokens".into(),
            ));
        }
        let rewind = self.offset - offset;
        let (capacity, write_index) = if rewind == 0 || self.max_context.is_none() {
            (self.capacity, self.write_index)
        } else if offset >= self.prefix_floor() {
            // The chronological view keeps the first `capacity` tokens only, so
            // the rewound snapshot shares the arrays and drops their tail.
            (self.capacity - rewind, self.write_index - rewind)
        } else {
            return Err(Error::InvalidModel(format!(
                "KV cache cannot rewind from {} to {offset} tokens",
                self.offset
            )));
        };
        Ok(Self {
            history: None,
            keys: clone_array(self.keys.as_ref())?,
            values: clone_array(self.values.as_ref())?,
            pages: self.pages.as_ref().map(|pages| pages.snapshot_at(offset)).transpose()?,
            offset,
            capacity,
            write_index,
            step: self.step,
            max_context: self.max_context,
            reserve_tokens: self.reserve_tokens,
            last_update_tokens: self.last_update_tokens,
        })
    }

    /// Tokens of the sliding chronological view (`temporal` in `sliding`).
    const fn sliding_valid(&self, window: usize) -> usize {
        if self.offset < window {
            self.offset
        } else {
            self.capacity
        }
    }
}
