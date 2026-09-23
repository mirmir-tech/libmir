use runtime::kv::BlockHash;

use super::{MixedMixerExecution, required};
use crate::{
    Result, backend::SharedRoutedCheckpoint, engine::model::generation::GenerationExecution,
};

#[derive(Debug)]
struct Entry {
    hash: BlockHash,
    tokens: usize,
    checkpoint: SharedRoutedCheckpoint,
    used: u64,
}

/// Retained prefix states, bounded by entry count and by device bytes. A
/// checkpoint holds the recurrent state of every linear layer, 63 MiB for
/// Qwen3.6-35B-A3B but 151 MiB for Qwen3.8-27B, so a count alone let the
/// cache grow past the host memory left beside a large model.
pub(super) struct PrefixCheckpoints {
    entries: Vec<Entry>,
    capacity: usize,
    byte_budget: usize,
    bytes: usize,
    clock: u64,
}

impl PrefixCheckpoints {
    pub(super) fn new(capacity: usize, byte_budget: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            byte_budget,
            bytes: 0,
            clock: 0,
        }
    }

    fn evict_oldest(&mut self) {
        let Some(oldest) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.used)
            .map(|(index, _)| index)
        else {
            return;
        };
        let removed = self.entries.swap_remove(oldest);
        self.bytes = self.bytes.saturating_sub(removed.checkpoint.bytes());
    }

    pub(super) fn lookup(
        &mut self,
        model: &str,
        prompt: &[u32],
        minimum: usize,
        maximum: usize,
    ) -> Option<&SharedRoutedCheckpoint> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.tokens > minimum
                    && entry.tokens <= maximum
                    && entry.tokens <= prompt.len()
                    && entry.hash == key(model, prompt, entry.tokens)
            })
            .max_by_key(|(_, entry)| entry.tokens)
            .map(|(index, _)| index)?;
        self.clock = self.clock.wrapping_add(1);
        self.entries[index].used = self.clock;
        Some(&self.entries[index].checkpoint)
    }

    pub(super) fn insert(
        &mut self,
        model: &str,
        prompt: &[u32],
        tokens: usize,
        checkpoint: SharedRoutedCheckpoint,
    ) {
        let bytes = checkpoint.bytes();
        if self.capacity == 0 || bytes > self.byte_budget || tokens == 0 || tokens > prompt.len() {
            return;
        }
        let hash = key(model, prompt, tokens);
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.tokens == tokens && entry.hash == hash)
        {
            self.bytes = self.bytes.saturating_sub(entry.checkpoint.bytes()).saturating_add(bytes);
            *entry = Entry {
                hash,
                tokens,
                checkpoint,
                used: self.clock,
            };
            return;
        }
        while self.entries.len() >= self.capacity
            || self.bytes.saturating_add(bytes) > self.byte_budget
        {
            self.evict_oldest();
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.entries.push(Entry {
            hash,
            tokens,
            checkpoint,
            used: self.clock,
        });
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

fn key(model: &str, prompt: &[u32], tokens: usize) -> BlockHash {
    BlockHash::from_tokens(model, None, &prompt[..tokens])
}

impl MixedMixerExecution {
    /// Arms the terminal checkpoint when the coming chunk runs through it, so
    /// the short prompt tail behind it needs no forward pass of its own.
    pub(super) fn arm_terminal_checkpoint(
        &mut self,
        request: &runtime::backend::PrefillRequest,
        offset: usize,
        count: usize,
    ) -> Result<()> {
        let Some(terminal) = self.terminal_cache_checkpoint(request) else {
            return Ok(());
        };
        if terminal > offset && terminal - offset < count {
            required(&mut self.sessions, request.session_id)?.arm_checkpoint(terminal - offset);
        }
        Ok(())
    }

    pub(super) fn checkpoint_prefix(
        &mut self,
        request: &runtime::backend::PrefillRequest,
    ) -> Result<()> {
        let terminal = request.terminal_cache_checkpoint();
        let session = required(&mut self.sessions, request.session_id)?;
        let position = session.position();
        let staged = session.take_staged_checkpoint(terminal.unwrap_or_default())?;
        let declared = request.cache_checkpoints.binary_search(&position).is_ok();
        let reached = (declared || terminal == Some(position))
            .then(|| session.checkpoint().map(|checkpoint| (position, checkpoint)))
            .transpose()?;
        for (position, checkpoint) in terminal.zip(staged).into_iter().chain(reached) {
            let bytes = checkpoint.bytes();
            self.checkpoints.insert(
                &request.model.id,
                &request.prompt_tokens,
                position,
                checkpoint,
            );
            tracing::debug!(
                backend = "cuda",
                session = %request.session_id,
                prefix_checkpoint_tokens = position,
                checkpoint_bytes = bytes,
                "retained CUDA mixed-mixer prefix checkpoint"
            );
        }
        Ok(())
    }
}
