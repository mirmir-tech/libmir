use std::ops::Range;

use runtime::kv::{CacheConfig, KvBlockWrite};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RingGeometry {
    blocks_per_session: usize,
    sessions: usize,
}

impl RingGeometry {
    pub(super) fn new(window: usize, cache: CacheConfig, sessions: usize) -> Result<Self> {
        if window == 0 || cache.block_size == 0 || sessions == 0 {
            return Err(Error::InvalidPagedKv("windowed KV ring geometry is empty"));
        }
        let blocks_per_session = window
            .checked_add(cache.block_size - 1)
            .ok_or(Error::InvalidPagedKv("windowed KV ring size overflow"))?
            .div_ceil(cache.block_size);
        Ok(Self { blocks_per_session, sessions })
    }

    pub(super) fn physical_blocks(self) -> Result<usize> {
        self.blocks_per_session
            .checked_mul(self.sessions)
            .ok_or(Error::InvalidPagedKv("windowed KV ring capacity overflow"))
    }

    pub(super) fn block(self, session_slot: usize, logical_block: usize) -> Result<usize> {
        if session_slot >= self.sessions {
            return Err(Error::InvalidPagedKv("windowed KV session slot is out of bounds"));
        }
        Ok(session_slot * self.blocks_per_session + logical_block % self.blocks_per_session)
    }

    pub(super) fn write_block(self, session_slot: usize, write: KvBlockWrite) -> Result<usize> {
        self.block(session_slot, write.table_index)
    }

    pub(super) fn slot_blocks(self, session_slot: usize) -> Result<Range<usize>> {
        if session_slot >= self.sessions {
            return Err(Error::InvalidPagedKv("windowed KV session slot is out of bounds"));
        }
        let start = session_slot
            .checked_mul(self.blocks_per_session)
            .ok_or(Error::InvalidPagedKv("windowed KV session offset overflow"))?;
        Ok(start..start + self.blocks_per_session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_an_extra_page_for_unaligned_windows() -> Result<()> {
        let ring = RingGeometry::new(128, CacheConfig::new(1_024), 16)?;

        assert_eq!(ring.physical_blocks()?, 144);
        assert_eq!(ring.block(3, 8)?, 35);
        assert_eq!(ring.block(3, 9)?, 27);
        assert_eq!(ring.slot_blocks(3)?, 27..36);
        Ok(())
    }
}
