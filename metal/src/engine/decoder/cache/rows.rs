//! Prefixes of a cache: how far its snapshots rewind, and verification
//! passes that keep a prefix of their rows, where recurrent layers record
//! the state after each row and attention K/V rewinds to the kept offset.

use super::{CacheStorage, DecoderCache, hybrid::HybridLinearLayerCache};
use crate::engine::{Error, KvCache, Result, Stream};

impl DecoderCache {
    /// The smallest prefix length a snapshot of this cache restores exactly:
    /// the deepest any layer can rewind. Recurrent state never rewinds.
    pub(crate) fn prefix_floor(&self) -> Result<usize> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                Ok(caches.iter().map(KvCache::prefix_floor).max().unwrap_or(0))
            },
            CacheStorage::HybridLinear(_) => self.cached_tokens(),
        }
    }

    /// Records the next pass's per-row recurrent states for
    /// [`Self::commit_rows`].
    pub(crate) fn arm_row_trail(&mut self) -> Result<()> {
        let CacheStorage::HybridLinear(layers) = &mut self.storage else {
            return Ok(());
        };
        layers.iter_mut().try_for_each(|layer| match layer {
            HybridLinearLayerCache::Linear(state) => {
                state.arm_row_trail();
                Ok(())
            },
            HybridLinearLayerCache::Full(_) => Ok(()),
            HybridLinearLayerCache::StateSpace(_) => {
                Err(Error::InvalidModel("state-space layers do not record per-row states".into()))
            },
        })
    }

    /// Keeps the first `rows` rows of the pass that started at `start`.
    pub(crate) fn commit_rows(&mut self, start: usize, rows: usize, stream: &Stream) -> Result<()> {
        let offset = start + rows;
        match &mut self.storage {
            CacheStorage::Attention(caches) => {
                for cache in caches {
                    *cache = cache.snapshot_at(offset)?;
                }
            },
            CacheStorage::HybridLinear(layers) => {
                for layer in layers {
                    match layer {
                        HybridLinearLayerCache::Linear(state) => state.commit_rows(rows, stream)?,
                        HybridLinearLayerCache::Full(cache) => {
                            *cache = cache.snapshot_at(offset)?;
                        },
                        HybridLinearLayerCache::StateSpace(_) => {
                            return Err(Error::InvalidModel(
                                "state-space layers cannot commit a row prefix".into(),
                            ));
                        },
                    }
                }
            },
        }
        Ok(())
    }
}
