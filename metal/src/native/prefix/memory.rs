use std::collections::HashSet;

use super::{PrefixCache, PrefixSnapshot};
use crate::{
    engine::{Array, DecoderCache},
    native::{error::Result, session::SessionState},
};

impl PrefixSnapshot {
    pub(super) fn new(cache: DecoderCache, position: usize, logits: Option<Array>) -> Result<Self> {
        // These handles retain available buffers only, never graphs. Allocation
        // identity accounts for row views, convolution tails and shared snapshots.
        let recurrent = cache.recurrent_allocations()?;
        Ok(Self {
            state: SessionState::from_prefix(cache, position),
            logits,
            recurrent,
        })
    }
}

impl PrefixCache {
    #[cfg(test)]
    pub(in crate::native) fn convolution_bytes(&self) -> Result<usize> {
        let mut allocations = HashSet::new();
        for snapshot in self
            .groups
            .values()
            .flat_map(|group| group.terminal.iter().chain(group.checkpoints.values()))
        {
            allocations.extend(snapshot.state.cache.convolution_allocations()?);
        }
        Ok(allocations.iter().map(mirtal::memory::Allocation::bytes).sum())
    }

    pub(in crate::native) fn recurrent_bytes(&self) -> usize {
        self.groups
            .values()
            .flat_map(|group| group.terminal.iter().chain(group.checkpoints.values()))
            .flat_map(|snapshot| &snapshot.recurrent)
            .collect::<HashSet<_>>()
            .into_iter()
            .fold(0_usize, |bytes, allocation| bytes.saturating_add(allocation.bytes()))
    }
}
