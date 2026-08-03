use super::PagedStore;

impl PagedStore {
    pub(in crate::engine::kv) fn page_count(&self) -> usize {
        self.storage.as_ref().map_or(0, |storage| storage.page_ids.len())
    }

    pub(crate) fn shares_arena(&self, other: &Self) -> bool {
        self.storage
            .as_ref()
            .zip(other.storage.as_ref())
            .is_some_and(|(left, right)| super::pool::same(&left.arena, &right.arena))
    }

    pub(crate) fn first_page(&self) -> Option<u32> {
        self.storage.as_ref().and_then(|storage| storage.page_ids.first().copied())
    }
}
