use super::KvCache;
use crate::error::{Result, RuntimeError};

impl KvCache {
    pub(super) fn ensure_free_blocks(&mut self, count: usize) -> Result<()> {
        if count > self.blocks.len() {
            return Err(RuntimeError::KvCapacity {
                requested: count,
                capacity: self.blocks.len(),
            });
        }
        if count <= self.free.len() {
            return Ok(());
        }
        for hash in self.prefix.oldest_first() {
            if count <= self.free.len() {
                break;
            }
            let Some(block) = self.prefix.peek(hash) else {
                continue;
            };
            if self.block(block)?.ref_count == 1 {
                let _evicted = self.evict_prefix(hash)?;
            } else {
                self.counters.protected_prefix_skips += 1;
            }
        }
        if count <= self.free.len() {
            Ok(())
        } else {
            Err(RuntimeError::KvCachePressure)
        }
    }
}
