mod index;

use std::collections::{HashMap, HashSet, VecDeque};

use self::index::{PrefixKey, indexed_prefixes, longest_indexed_prefix};
use super::{error::Result, session::SessionState};
use crate::engine::Array;

#[derive(Debug, Clone, Copy)]
struct PrefixEntry {
    memory_group: u64,
    position: usize,
}

#[derive(Debug)]
struct PrefixSnapshot {
    state: SessionState,
    logits: Array,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct PrefixCache {
    entries: HashMap<PrefixKey, PrefixEntry>,
    groups: HashMap<u64, PrefixSnapshot>,
    group_recency: VecDeque<u64>,
    capacity: usize,
    byte_capacity: usize,
    next_memory_group: u64,
}

impl PrefixCache {
    pub(super) fn new(capacity: usize, byte_capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            groups: HashMap::new(),
            group_recency: VecDeque::new(),
            capacity,
            byte_capacity,
            next_memory_group: 0,
        }
    }

    pub(super) fn restore_longest(
        &mut self,
        model: &str,
        tokens: &[u32],
    ) -> Result<Option<(SessionState, Option<Array>)>> {
        let Some((_, entry)) = longest_indexed_prefix(model, tokens, &self.entries) else {
            self.reserve_miss_slot();
            return Ok(None);
        };
        let group = entry.memory_group;
        let Some(snapshot) = self.groups.get(&group) else {
            return Ok(None);
        };
        let exact = entry.position == tokens.len() && entry.position == snapshot.state.position;
        let position = if entry.position == tokens.len() && !exact {
            entry.position.saturating_sub(1)
        } else {
            entry.position
        };
        let cache = snapshot.state.cache.snapshot_at(position)?;
        let logits = exact.then(|| snapshot.logits.snapshot()).transpose()?;
        self.touch_group(group);
        Ok(Some((SessionState::from_prefix(cache, position), logits)))
    }

    pub(super) fn reserve_batch_slots(&mut self, count: usize) -> bool {
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

    pub(super) fn insert(
        &mut self,
        model: &str,
        tokens: &[u32],
        state: &SessionState,
        logits: &Array,
        block_size: Option<usize>,
        bytes: usize,
    ) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let memory_group = self.next_memory_group;
        self.next_memory_group = self.next_memory_group.wrapping_add(1);
        let snapshot = PrefixSnapshot {
            state: SessionState::from_prefix(
                state.cache.snapshot_at(state.position)?,
                state.position,
            ),
            logits: logits.snapshot()?,
            bytes,
        };
        let block_size =
            block_size.filter(|size| *size > 0 && state.cache.supports_prefix_offsets());
        for (key, position) in indexed_prefixes(model, tokens, block_size) {
            self.entries.insert(key, PrefixEntry { memory_group, position });
        }
        self.groups.insert(memory_group, snapshot);
        self.touch_group(memory_group);
        self.remove_unindexed_groups();
        self.enforce_limits();
        Ok(())
    }

    fn enforce_limits(&mut self) {
        while self.groups.len() > self.capacity || self.resident_bytes() > self.byte_capacity {
            if !self.evict_oldest() {
                break;
            }
        }
    }

    pub(super) const fn enabled(&self) -> bool {
        self.capacity > 0 && self.byte_capacity > 0
    }

    pub(super) const fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) const fn byte_capacity(&self) -> usize {
        self.byte_capacity
    }

    pub(super) fn resident_bytes(&self) -> usize {
        self.groups.values().map(|group| group.bytes).sum()
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.groups.clear();
        self.group_recency.clear();
    }

    pub(super) fn evict_oldest(&mut self) -> bool {
        let Some(expired) = self.group_recency.pop_front() else {
            return false;
        };
        self.groups.remove(&expired);
        self.entries.retain(|_, entry| entry.memory_group != expired);
        true
    }

    fn remove_unindexed_groups(&mut self) {
        let indexed = self.entries.values().map(|entry| entry.memory_group).collect::<HashSet<_>>();
        self.groups.retain(|group, _| indexed.contains(group));
        self.group_recency.retain(|group| indexed.contains(group));
    }

    fn reserve_miss_slot(&mut self) {
        if self.groups.len() >= self.capacity || self.resident_bytes() >= self.byte_capacity {
            self.evict_oldest();
        }
    }

    fn touch_group(&mut self, group: u64) {
        self.remove_group_recency(group);
        self.group_recency.push_back(group);
    }

    fn remove_group_recency(&mut self, group: u64) {
        if let Some(index) = self.group_recency.iter().position(|entry| *entry == group) {
            let _removed = self.group_recency.remove(index);
        }
    }
}

#[cfg(test)]
mod tests;
