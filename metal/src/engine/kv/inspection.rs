use super::{KvCache, KvStorage};

impl KvCache {
    pub(crate) fn extend_history_allocations(
        &self,
        allocations: &mut std::collections::HashSet<mirtal::memory::Allocation>,
    ) -> super::Result<()> {
        if let Some(row) = &self.history {
            row.extend_allocations(allocations)?;
        }
        Ok(())
    }

    pub(crate) fn has_native_page_views(&self) -> bool {
        match &self.storage {
            KvStorage::Paged(pages) => pages.active() && !pages.quantized(),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => false,
        }
    }

    pub(crate) fn shares_paged_arena(&self, other: &Self) -> bool {
        match (&self.storage, &other.storage) {
            (KvStorage::Paged(left), KvStorage::Paged(right)) => left.shares_arena(right),
            _ => false,
        }
    }

    pub(crate) fn first_physical_page(&self) -> Option<u32> {
        match &self.storage {
            KvStorage::Paged(pages) => pages.first_page(),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => None,
        }
    }
}
