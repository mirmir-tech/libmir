use std::collections::HashSet;

use super::PrefixCache;

impl PrefixCache {
    pub(in crate::native) fn reserve_batch_slots(&mut self, count: usize) -> bool {
        let target = self.capacity.saturating_sub(count);
        let mut evicted = false;
        while self.groups.len() > target {
            if !self.evict_oldest() {
                break;
            }
            evicted = true;
        }
        evicted
    }

    pub(super) fn enforce_limits(&mut self) {
        while self.groups.len() > self.capacity || self.resident_bytes() > self.byte_capacity {
            if !self.evict_oldest() {
                break;
            }
        }
    }

    pub(in crate::native) const fn enabled(&self) -> bool {
        self.capacity > 0 && self.byte_capacity > 0
    }

    pub(in crate::native) const fn capacity(&self) -> usize {
        self.capacity
    }

    pub(in crate::native) const fn byte_capacity(&self) -> usize {
        self.byte_capacity
    }

    pub(in crate::native) fn resident_bytes(&self) -> usize {
        self.groups.values().map(|group| group.bytes).sum()
    }

    pub(in crate::native) fn clear(&mut self) {
        self.entries.clear();
        self.groups.clear();
        self.group_recency.clear();
    }

    pub(in crate::native) fn evict_oldest(&mut self) -> bool {
        let Some(expired) = self.group_recency.pop_front() else {
            return false;
        };
        self.groups.remove(&expired);
        self.entries.retain(|_, entry| entry.memory_group != expired);
        true
    }

    pub(super) fn remove_unindexed_groups(&mut self) {
        let indexed = self.entries.values().map(|entry| entry.memory_group).collect::<HashSet<_>>();
        self.groups.retain(|group, _| indexed.contains(group));
        self.group_recency.retain(|group| indexed.contains(group));
    }

    pub(super) fn reserve_miss_slot(&mut self) {
        if self.groups.len() >= self.capacity || self.resident_bytes() >= self.byte_capacity {
            self.evict_oldest();
        }
    }

    pub(super) fn touch_group(&mut self, group: u64) {
        if let Some(index) = self.group_recency.iter().position(|entry| *entry == group) {
            let _removed = self.group_recency.remove(index);
        }
        self.group_recency.push_back(group);
    }
}
