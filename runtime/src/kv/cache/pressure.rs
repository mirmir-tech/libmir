use super::{KvCache, KvResidency};
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

    /// Under [`KvResidency::OnDemand`], evicts the oldest unshared cached
    /// prefixes until `count` consecutive blocks are free below the resident
    /// ones, while those prefixes hold more blocks than live requests:
    /// storage then grows for live requests and keeps cached prefixes as
    /// large as them. A prefix's blocks were taken in a run, so evicting the
    /// oldest frees runs.
    pub(super) fn reuse_resident_blocks(&mut self, count: usize) -> Result<()> {
        if self.residency != KvResidency::OnDemand {
            return Ok(());
        }
        let Ok(run) = u32::try_from(count) else {
            return Ok(());
        };
        if run == 0 || self.run_below_resident(run).is_some() {
            return Ok(());
        }
        let oldest = self.prefix.oldest_first();
        let mut idle = Vec::with_capacity(oldest.len());
        for hash in oldest {
            if let Some(block) = self.prefix.peek(hash)
                && self.block(block)?.ref_count == 1
            {
                idle.push(hash);
            }
        }
        let used = self.blocks.len() - self.free.len();
        let live = used - idle.len() + count;
        let mut cached = idle.len();
        for hash in idle {
            if cached <= live {
                break;
            }
            if self.evict_prefix(hash)? {
                cached -= 1;
                if self.run_below_resident(run).is_some() {
                    break;
                }
            }
        }
        Ok(())
    }
}
