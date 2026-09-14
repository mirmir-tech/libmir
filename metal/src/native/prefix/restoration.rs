use super::{LeasedPrefix, PrefixCache, index::longest_usable_prefix};
use crate::{
    engine::Array,
    native::{error::Result, session::SessionState},
};

impl PrefixCache {
    pub(in crate::native) fn lease_longest(
        &mut self,
        model: &str,
        tokens: &[u32],
    ) -> Result<Option<LeasedPrefix>> {
        let Some((_, entry)) = longest_usable_prefix(model, tokens, &self.entries, |entry| {
            entry.position < tokens.len()
                || self
                    .groups
                    .get(&entry.memory_group)
                    .and_then(|group| {
                        group.checkpoints.get(&entry.position).or(group.terminal.as_ref())
                    })
                    .is_some_and(|snapshot| {
                        snapshot.logits.is_some() || snapshot.state.cache.supports_prefix_offsets()
                    })
        }) else {
            return Ok(None);
        };
        let group = entry.memory_group;
        let Some(group_state) = self.groups.get(&group) else {
            return Ok(None);
        };
        let direct = group_state.checkpoints.get(&entry.position);
        let source = if let Some(checkpoint) = direct {
            checkpoint
        } else {
            let Some(terminal) = group_state.terminal.as_ref() else {
                return Ok(None);
            };
            terminal
        };
        let complete_prompt = entry.position == tokens.len();
        let exact =
            complete_prompt && entry.position == source.state.position && source.logits.is_some();
        let position = if exact {
            entry.position
        } else if complete_prompt {
            entry.completion_position
        } else {
            entry.continuation_position
        };
        let cache = source.state.cache.snapshot_at(position)?;
        let logits = if exact {
            source.logits.as_ref().map(Array::snapshot).transpose()?
        } else {
            None
        };
        self.touch_group(group);
        Ok(Some(LeasedPrefix {
            restored: (SessionState::from_prefix(cache, position), logits),
            memory_group: group,
        }))
    }
}
