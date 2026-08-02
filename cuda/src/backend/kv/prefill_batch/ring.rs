use runtime::kv::BlockTable;

use super::PagedPrefillBatch;
use crate::{Error, Result};

impl PagedPrefillBatch {
    pub(crate) fn prepare_ring(
        &mut self,
        tables: &[&BlockTable],
        starts: &[usize],
        query_tokens: &[usize],
        session_slots: &[usize],
        ring_blocks: usize,
        attention_window: usize,
    ) -> Result<()> {
        let retained_tokens = ring_blocks
            .checked_mul(self.cache.block_size)
            .ok_or(Error::InvalidPagedKv("windowed KV retained range overflow"))?;
        if ring_blocks == 0
            || attention_window == 0
            || retained_tokens < attention_window
            || session_slots.len() != tables.len()
            || tables.len() != starts.len()
            || tables.len() != query_tokens.len()
        {
            return Err(Error::InvalidPagedKv("invalid windowed prefill metadata geometry"));
        }
        self.ring_tables.host.fill(u32::MAX);
        self.ring_slot_mapping.host.fill(u32::MAX);
        let mut packed = 0;
        for (row, (((table, start), count), session_slot)) in
            tables.iter().zip(starts).zip(query_tokens).zip(session_slots).enumerate()
        {
            let base = session_slot
                .checked_mul(ring_blocks)
                .ok_or(Error::InvalidPagedKv("windowed KV session offset overflow"))?;
            let table_offset = row * self.max_blocks;
            for (logical, target) in self.ring_tables.host[table_offset..]
                .iter_mut()
                .take(table.blocks().len())
                .enumerate()
            {
                *target = u32::try_from(base + logical % ring_blocks)?;
            }
            let end = start
                .checked_add(*count)
                .ok_or(Error::InvalidPagedKv("windowed KV query range overflow"))?;
            for local in 0..*count {
                let position = start + local;
                if !retained(position, end, retained_tokens) {
                    continue;
                }
                let physical_block = base + (position / self.cache.block_size) % ring_blocks;
                let slot = physical_block
                    .checked_mul(self.cache.block_size)
                    .and_then(|value| value.checked_add(position % self.cache.block_size))
                    .ok_or(Error::InvalidPagedKv("windowed KV slot mapping overflow"))?;
                self.ring_slot_mapping.host[packed + local] = u32::try_from(slot)?;
            }
            packed += count;
        }
        self.ring_tables.upload(&self.stream)?;
        self.ring_slot_mapping.upload(&self.stream)
    }
}

const fn retained(position: usize, end: usize, ring_tokens: usize) -> bool {
    position >= end.saturating_sub(ring_tokens)
}

#[cfg(test)]
mod tests {
    use super::retained;

    #[test]
    fn remapping_formula_keeps_sessions_disjoint() {
        let ring_blocks = 9;
        let first = 2 * ring_blocks + 17 % ring_blocks;
        let second = 3 * ring_blocks + 17 % ring_blocks;

        assert_eq!(first, 26);
        assert_eq!(second, 35);
    }

    #[test]
    fn stores_only_the_latest_window_after_bulk_prefill() {
        assert!(!retained(8_047, 8_192, 144));
        assert!(retained(8_048, 8_192, 144));
        assert!(retained(8_191, 8_192, 128));
    }
}
