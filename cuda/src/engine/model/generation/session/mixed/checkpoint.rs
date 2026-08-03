use runtime::kv::BlockHash;

use crate::backend::SharedRoutedCheckpoint;

#[derive(Debug)]
struct Entry {
    hash: BlockHash,
    tokens: usize,
    checkpoint: SharedRoutedCheckpoint,
    used: u64,
}

pub(super) struct PrefixCheckpoints {
    entries: Vec<Entry>,
    capacity: usize,
    clock: u64,
}

impl PrefixCheckpoints {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            clock: 0,
        }
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
        if self.capacity == 0 || tokens == 0 || tokens > prompt.len() {
            return;
        }
        let hash = key(model, prompt, tokens);
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.tokens == tokens && entry.hash == hash)
        {
            *entry = Entry {
                hash,
                tokens,
                checkpoint,
                used: self.clock,
            };
            return;
        }
        if self.entries.len() == self.capacity {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(index, _)| index)
                .unwrap_or_default();
            self.entries.swap_remove(oldest);
        }
        self.entries.push(Entry {
            hash,
            tokens,
            checkpoint,
            used: self.clock,
        });
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }
}

fn key(model: &str, prompt: &[u32], tokens: usize) -> BlockHash {
    BlockHash::from_tokens(model, None, &prompt[..tokens])
}
