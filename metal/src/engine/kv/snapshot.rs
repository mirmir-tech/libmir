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

impl KvCache {
    /// Keeps a prefix snapshot's partial last page (`paged::tail`).
    pub(crate) fn capture_tail(&mut self, stream: &crate::engine::Stream) -> Result<()> {
        match &mut self.storage {
            KvStorage::Paged(pages) => pages.capture_tail(self.offset, stream),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => Ok(()),
        }
    }

    pub(crate) fn tail_roots(&self) -> Box<dyn Iterator<Item = &crate::engine::Array> + '_> {
        match &self.storage {
            KvStorage::Paged(pages) => Box::new(pages.tail_roots()),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => Box::new(std::iter::empty()),
        }
    }

    /// Whether a snapshot of this cache at `tokens` resumes from the
    /// runtime's first `cached` tokens; only pages live in the runtime's
    /// blocks.
    pub(crate) fn restorable_within(&self, tokens: usize, cached: usize) -> bool {
        match &self.storage {
            KvStorage::Paged(pages) => pages.restorable_within(tokens, cached),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => true,
        }
    }

    /// Writes a restored snapshot's partial last page into the blocks bound
    /// since.
    pub(crate) fn restore_tail(&mut self, stream: &crate::engine::Stream) -> Result<()> {
        match &mut self.storage {
            KvStorage::Paged(pages) => pages.restore_tail(self.offset, stream),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => Ok(()),
        }
    }
}
