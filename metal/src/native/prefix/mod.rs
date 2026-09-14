mod index;
mod memory;
mod restoration;
mod retention;

use std::collections::{HashMap, VecDeque};

use self::index::{PrefixKey, indexed_prefixes, longest_indexed_prefix};
use super::{
    error::{Error, Result},
    session::SessionState,
};
use crate::engine::Array;

pub(super) type RestoredPrefix = (SessionState, Option<Array>);

pub(super) struct LeasedPrefix {
    pub(super) restored: RestoredPrefix,
    pub(super) memory_group: u64,
}

#[derive(Debug, Clone, Copy)]
struct PrefixEntry {
    memory_group: u64,
    position: usize,
    continuation_position: usize,
    completion_position: usize,
}

#[derive(Debug)]
struct PrefixSnapshot {
    state: SessionState,
    logits: Option<Array>,
    recurrent: std::collections::HashSet<mirtal::memory::Allocation>,
}

#[derive(Debug, Default)]
struct PrefixGroup {
    terminal: Option<PrefixSnapshot>,
    checkpoints: HashMap<usize, PrefixSnapshot>,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct PrefixCache {
    entries: HashMap<PrefixKey, PrefixEntry>,
    groups: HashMap<u64, PrefixGroup>,
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
        let memory_group = self.pending_group(model, tokens).unwrap_or_else(|| self.new_group());
        let snapshot = PrefixSnapshot::new(
            state.cache.snapshot_at(state.position)?,
            state.position,
            Some(logits.snapshot()?),
        )?;
        let block_size =
            block_size.filter(|size| *size > 0 && state.cache.supports_prefix_offsets());
        for (key, position) in indexed_prefixes(model, tokens, block_size) {
            let continuation_position = block_size
                .filter(|size| *size > 0)
                .map_or(position, |size| position / size * size);
            let completion_position = block_size.filter(|size| *size > 0).map_or_else(
                || position.saturating_sub(1),
                |size| position.saturating_sub(1) / size * size,
            );
            self.entries.insert(
                key,
                PrefixEntry {
                    memory_group,
                    position,
                    continuation_position,
                    completion_position,
                },
            );
        }
        let group = self.groups.entry(memory_group).or_default();
        group.terminal = Some(snapshot);
        group.bytes = group.bytes.max(bytes);
        self.touch_group(memory_group);
        self.remove_unindexed_groups();
        self.enforce_limits();
        self.trace_snapshot("terminal", tokens.len());
        Ok(())
    }

    pub(super) fn insert_checkpoint(
        &mut self,
        model: &str,
        tokens: &[u32],
        state: &SessionState,
        block_size: usize,
        bytes: usize,
    ) -> Result<()> {
        if !self.enabled() || tokens.is_empty() {
            return Ok(());
        }
        let memory_group = self.pending_group(model, tokens).unwrap_or_else(|| self.new_group());
        let position = tokens.len();
        let continuation_position = if block_size > 0 && state.cache.supports_prefix_offsets() {
            position / block_size * block_size
        } else {
            position
        };
        let completion_position = if block_size > 0 && state.cache.supports_prefix_offsets() {
            position.saturating_sub(1) / block_size * block_size
        } else {
            position.saturating_sub(1)
        };
        let snapshot = PrefixSnapshot::new(state.cache.snapshot_at(position)?, position, None)?;
        let key = indexed_prefixes(model, tokens, None)
            .pop()
            .map(|(key, _)| key)
            .ok_or_else(|| Error::InvalidPrefillBatch("prefix checkpoint has no key".into()))?;
        self.entries.insert(
            key,
            PrefixEntry {
                memory_group,
                position,
                continuation_position,
                completion_position,
            },
        );
        let group = self.groups.entry(memory_group).or_default();
        group.checkpoints.insert(position, snapshot);
        group.bytes = group.bytes.max(bytes);
        self.touch_group(memory_group);
        self.remove_unindexed_groups();
        self.enforce_limits();
        self.trace_snapshot("checkpoint", tokens.len());
        Ok(())
    }

    fn pending_group(&self, model: &str, tokens: &[u32]) -> Option<u64> {
        longest_indexed_prefix(model, tokens, &self.entries)
            .map(|(_, entry)| entry.memory_group)
            .filter(|group| self.groups.get(group).is_some_and(|state| state.terminal.is_none()))
    }

    fn new_group(&mut self) -> u64 {
        let group = self.next_memory_group;
        self.next_memory_group = self.next_memory_group.wrapping_add(1);
        group
    }
}

#[cfg(test)]
mod tests;
