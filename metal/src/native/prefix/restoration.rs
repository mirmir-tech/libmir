use foundation::model::ModelId;

use super::{LeasedPrefix, PrefixCache, PrefixEntry, PrefixSnapshot, index::longest_usable_prefix};
use crate::{
    engine::Array,
    native::{error::Result, session::SessionState},
};

impl PrefixCache {
    /// [`Self::lease_longest_within`] without a block table, for tests.
    #[cfg(test)]
    pub(in crate::native) fn lease_longest(
        &mut self,
        model: &ModelId,
        tokens: &[u32],
    ) -> Result<Option<LeasedPrefix>> {
        self.lease_longest_within(model, tokens, None)
    }

    /// The longest snapshot of `tokens`'s prefix that resumes from the
    /// runtime's first `cached` tokens, when its block table backs the pages.
    pub(in crate::native) fn lease_longest_within(
        &mut self,
        model: &ModelId,
        tokens: &[u32],
        cached: Option<usize>,
    ) -> Result<Option<LeasedPrefix>> {
        let Some((_, entry)) = longest_usable_prefix(model, tokens, &self.entries, |entry| {
            self.source(entry).is_some_and(|source| {
                let (position, _) = resumption(entry, source, tokens.len());
                (entry.position < tokens.len()
                    || source.logits.is_some()
                    || source
                        .state
                        .cache
                        .prefix_floor()
                        .is_ok_and(|floor| floor <= entry.completion_position))
                    && cached
                        .is_none_or(|cached| source.state.cache.restorable_within(position, cached))
            })
        }) else {
            return Ok(None);
        };
        let Some(source) = self.source(&entry) else {
            return Ok(None);
        };
        let (position, exact) = resumption(&entry, source, tokens.len());
        let cache = source.state.cache.snapshot_at(position)?;
        let logits = if exact {
            source.logits.as_ref().map(Array::snapshot).transpose()?
        } else {
            None
        };
        let group = entry.memory_group;
        self.touch_group(group);
        Ok(Some(LeasedPrefix {
            restored: (SessionState::from_prefix(cache, position), logits),
            memory_group: group,
        }))
    }

    fn source(&self, entry: &PrefixEntry) -> Option<&PrefixSnapshot> {
        let group = self.groups.get(&entry.memory_group)?;
        group.checkpoints.get(&entry.position).or(group.terminal.as_ref())
    }
}

/// Where a request for a prompt of `prompt_len` tokens resumes from `entry`,
/// and whether exactly there with the snapshot's logits: a complete prompt
/// without them recomputes its last token, a continuation resumes at a block.
fn resumption(entry: &PrefixEntry, source: &PrefixSnapshot, prompt_len: usize) -> (usize, bool) {
    let complete_prompt = entry.position == prompt_len;
    let exact =
        complete_prompt && entry.position == source.state.position && source.logits.is_some();
    let position = if exact {
        entry.position
    } else if complete_prompt {
        entry.completion_position
    } else {
        entry.continuation_position
    };
    (position, exact)
}
