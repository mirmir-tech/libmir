use super::PagedPrefillBatch;
use crate::{Error, Result};

impl PagedPrefillBatch {
    pub(crate) const fn row_capacity(&self) -> usize {
        self.max_batch
    }

    pub(crate) const fn token_capacity(&self) -> usize {
        self.max_tokens
    }

    pub(crate) const fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    pub(crate) const fn token_counts(&self) -> &mircuda::DeviceBuffer<u32> {
        &self.token_counts.device
    }

    pub(crate) const fn ring_tables(&self) -> &mircuda::DeviceBuffer<u32> {
        &self.ring_tables.device
    }

    pub(crate) fn skip_cached_slot_writes(&mut self, write_starts: &[usize]) -> Result<()> {
        if write_starts.len() != self.active {
            return Err(Error::InvalidPagedKv(
                "cached prefill write offsets differ from batch rows",
            ));
        }
        let mut packed = 0;
        let mut changed = false;
        for (row, write_start) in self.rows.iter().zip(write_starts) {
            for local in 0..row.tokens() {
                if row.start() + local < *write_start {
                    self.slot_mapping.host[packed + local] = u32::MAX;
                    changed = true;
                }
            }
            packed += row.tokens();
        }
        if changed {
            self.slot_mapping.upload(&self.stream)?;
        }
        Ok(())
    }
}
