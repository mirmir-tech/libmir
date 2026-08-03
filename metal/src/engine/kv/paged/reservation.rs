use super::{Arena, Result, Storage};

impl Storage {
    pub(super) fn reservation_needed(&self, planned: usize) -> usize {
        let owned = self.page_ids.len() + self.reserved_page_ids.len();
        planned.saturating_sub(owned)
    }

    pub(super) fn additional_owned_pages(
        &self,
        planned: usize,
        needed: usize,
        shared: usize,
    ) -> usize {
        let owned = self.page_ids.len() + self.reserved_page_ids.len();
        planned.max(needed).saturating_sub(owned).saturating_add(shared)
    }

    pub(super) fn reserve_contiguous(&mut self, arena: &mut Arena, planned: usize) -> Result<()> {
        let count = self.reservation_needed(planned);
        if count > 0 {
            self.reserved_page_ids.extend(arena.allocate_contiguous(count)?);
        }
        Ok(())
    }

    pub(super) fn append_pages(&mut self, arena: &mut Arena, count: usize) -> Result<()> {
        let reserved = count.min(self.reserved_page_ids.len());
        self.page_ids.extend(self.reserved_page_ids.drain(..reserved));
        if reserved < count {
            self.page_ids.extend(arena.allocate_contiguous(count - reserved)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::engine::Array;

    #[test]
    fn counts_only_unowned_planned_pages() -> Result<()> {
        let arena = Arc::new(Mutex::new(Arena {
            keys: Array::from_f32(&[0.0], &[1, 1, 1, 1])?,
            values: Array::from_f32(&[0.0], &[1, 1, 1, 1])?,
            key_scales: None,
            value_scales: None,
            capacity: 1,
            page_size: 1,
            kv_heads: 1,
            head_dim: 1,
            references: vec![1],
        }));
        let storage = Storage {
            arena,
            table: Array::from_u32(&[0], &[1])?,
            page_ids: vec![0],
            reserved_page_ids: vec![1, 2],
            table_capacity: 1,
            identity: true,
        };
        assert_eq!(storage.additional_owned_pages(5, 4, 1), 3);
        assert_eq!(storage.reservation_needed(5), 2);
        Ok(())
    }
}
