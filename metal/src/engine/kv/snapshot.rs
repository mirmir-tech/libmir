//! Prefix snapshots of one layer's K/V, at its current offset or rewound.

use super::{ContiguousStore, KvCache, KvStorage, SlidingStore, clone_array};
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
        match &self.storage {
            KvStorage::Paged(_) => 0,
            KvStorage::Contiguous(_) => self.offset,
            KvStorage::Sliding(store) => {
                if store.keys.is_none() {
                    return self.offset;
                }
                let stored = store.valid_tokens(self.offset);
                if store.write_index < stored {
                    self.offset
                } else if stored >= self.offset {
                    0
                } else {
                    (self.offset - stored).saturating_add(store.max_context - 1).min(self.offset)
                }
            },
        }
    }

    pub fn snapshot_at(&self, offset: usize) -> Result<Self> {
        if offset > self.offset {
            return Err(Error::InvalidModel(
                "KV cache snapshot offset exceeds cached tokens".into(),
            ));
        }
        let rewind = self.offset - offset;
        let storage = match &self.storage {
            KvStorage::Contiguous(store) => KvStorage::Contiguous(ContiguousStore {
                keys: clone_array(store.keys.as_ref())?,
                values: clone_array(store.values.as_ref())?,
                capacity: store.capacity,
                step: store.step,
            }),
            KvStorage::Sliding(store) => {
                let (capacity, write_index) = if rewind == 0 {
                    (store.capacity, store.write_index)
                } else if offset >= self.prefix_floor() {
                    (store.capacity - rewind, store.write_index - rewind)
                } else {
                    return Err(Error::InvalidModel(format!(
                        "KV cache cannot rewind from {} to {offset} tokens",
                        self.offset
                    )));
                };
                KvStorage::Sliding(SlidingStore {
                    keys: clone_array(store.keys.as_ref())?,
                    values: clone_array(store.values.as_ref())?,
                    capacity,
                    write_index,
                    step: store.step,
                    max_context: store.max_context,
                    last_update_tokens: store.last_update_tokens,
                })
            },
            KvStorage::Paged(pages) => KvStorage::Paged(pages.snapshot_at(offset)?),
        };
        Ok(Self { history: None, storage, offset })
    }
}
