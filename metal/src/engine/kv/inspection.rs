use super::KvCache;

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
        self.pages.as_ref().is_some_and(|pages| pages.active() && !pages.quantized())
    }

    pub(crate) fn shares_paged_arena(&self, other: &Self) -> bool {
        self.pages
            .as_ref()
            .zip(other.pages.as_ref())
            .is_some_and(|(left, right)| left.shares_arena(right))
    }

    pub(crate) fn first_physical_page(&self) -> Option<u32> {
        self.pages.as_ref().and_then(super::paged::PagedStore::first_page)
    }
}
