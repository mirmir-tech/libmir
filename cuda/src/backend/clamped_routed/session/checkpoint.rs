use runtime::kv::BlockHash;
use uuid::Uuid;

use super::CudaClampedRoutedModelSession;
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CheckpointKey {
    hash: BlockHash,
    tokens: usize,
}

#[derive(Clone, Copy, Debug)]
struct Checkpoint {
    key: CheckpointKey,
    slot: usize,
    used: u64,
}

#[derive(Debug)]
pub(super) struct PrefixCheckpoints {
    entries: Vec<Checkpoint>,
    free: Vec<usize>,
    clock: u64,
}

impl PrefixCheckpoints {
    pub(super) fn new(first_slot: usize, capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free: (first_slot..first_slot.saturating_add(capacity)).rev().collect(),
            clock: 0,
        }
    }

    pub(super) fn lookup(
        &mut self,
        model: &str,
        prompt: &[u32],
        minimum: usize,
        maximum: usize,
    ) -> Option<(usize, usize)> {
        let mut lengths = self
            .entries
            .iter()
            .map(|entry| entry.key.tokens)
            .filter(|tokens| *tokens > minimum && *tokens <= maximum && *tokens <= prompt.len())
            .collect::<Vec<_>>();
        lengths.sort_unstable_by(|left, right| right.cmp(left));
        lengths.dedup();
        for tokens in lengths {
            let key = key(model, prompt, tokens);
            if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
                let used = self.tick();
                self.entries[index].used = used;
                return Some((tokens, self.entries[index].slot));
            }
        }
        None
    }

    pub(super) fn slot_for_write(
        &mut self,
        model: &str,
        prompt: &[u32],
        tokens: usize,
    ) -> (BlockHash, usize) {
        let key = key(model, prompt, tokens);
        let used = self.tick();
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
            entry.used = used;
            return (key.hash, entry.slot);
        }
        let slot = self.free.pop().unwrap_or_else(|| {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(index, _)| index)
                .unwrap_or_default();
            self.entries.swap_remove(oldest).slot
        });
        self.entries.push(Checkpoint { key, slot, used });
        (key.hash, slot)
    }

    pub(super) fn invalidate(&mut self, hash: BlockHash, tokens: usize) {
        let key = CheckpointKey { hash, tokens };
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            let entry = self.entries.swap_remove(index);
            self.free.push(entry.slot);
        }
    }

    pub(super) fn clear(&mut self) {
        self.free.extend(self.entries.drain(..).map(|entry| entry.slot));
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }
}

impl CudaClampedRoutedModelSession {
    pub(crate) fn checkpoint_prefix(
        &mut self,
        session: Uuid,
        model: &str,
        prompt: &[u32],
        end: usize,
    ) -> Result<()> {
        let block_size = self.template.cache.block_size;
        let tokens = checkpoint_tokens(end, block_size);
        if tokens == 0 || tokens > prompt.len() {
            return Ok(());
        }
        let active_slot = self.rings.slot(session)?;
        let (hash, checkpoint_slot) = self.checkpoints.slot_for_write(model, prompt, tokens);
        for cache in self.state.caches.iter_mut().filter(|cache| cache.is_windowed()) {
            if let Err(error) = cache.copy_ring_slot(active_slot, checkpoint_slot) {
                self.checkpoints.invalidate(hash, tokens);
                return Err(error);
            }
        }
        tracing::debug!(
            backend = "cuda",
            %session,
            prefix_checkpoint_tokens = tokens,
            checkpoint_slot,
            "retained CUDA sliding-window prefix checkpoint"
        );
        Ok(())
    }

    pub(crate) fn restore_prefix(
        &mut self,
        session: Uuid,
        model: &str,
        prompt: &[u32],
        minimum: usize,
        maximum: usize,
    ) -> Result<Option<usize>> {
        let Some((tokens, checkpoint_slot)) =
            self.checkpoints.lookup(model, prompt, minimum, maximum)
        else {
            return Ok(None);
        };
        let active_slot = self.rings.acquire(session)?;
        for cache in self.state.caches.iter_mut().filter(|cache| cache.is_windowed()) {
            cache.copy_ring_slot(checkpoint_slot, active_slot)?;
        }
        self.positions.insert(session, tokens);
        tracing::debug!(
            backend = "cuda",
            %session,
            prefix_checkpoint_tokens = tokens,
            checkpoint_slot,
            active_slot,
            "restored CUDA sliding-window prefix checkpoint"
        );
        Ok(Some(tokens))
    }
}

fn checkpoint_tokens(end: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return 0;
    }
    end.saturating_sub(block_size) / block_size * block_size
}

fn key(model: &str, prompt: &[u32], tokens: usize) -> CheckpointKey {
    CheckpointKey {
        hash: BlockHash::from_tokens(model, None, &prompt[..tokens]),
        tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_retained_prefixes_to_complete_cache_blocks() {
        assert_eq!(checkpoint_tokens(2_048, 16), 2_032);
        assert_eq!(checkpoint_tokens(4_097, 16), 4_080);
        assert_eq!(checkpoint_tokens(8_193, 16), 8_176);
        assert_eq!(checkpoint_tokens(15, 16), 0);
        assert_eq!(checkpoint_tokens(64, 0), 0);
    }

    #[test]
    fn returns_the_longest_matching_checkpoint_above_replay() {
        let prompt = (0..64).collect::<Vec<_>>();
        let mut cache = PrefixCheckpoints::new(4, 3);
        cache.slot_for_write("model", &prompt, 16);
        cache.slot_for_write("model", &prompt, 32);
        cache.slot_for_write("model", &prompt, 48);

        assert_eq!(cache.lookup("model", &prompt, 20, 48), Some((48, 6)));
        let mut changed = prompt.clone();
        changed[40] = 7;
        assert_eq!(cache.lookup("model", &changed, 20, 48), Some((32, 5)));
    }

    #[test]
    fn evicts_the_least_recent_checkpoint() {
        let prompt = (0..64).collect::<Vec<_>>();
        let mut cache = PrefixCheckpoints::new(8, 2);
        cache.slot_for_write("model", &prompt, 16);
        cache.slot_for_write("model", &prompt, 32);
        assert_eq!(cache.lookup("model", &prompt, 0, 16), Some((16, 8)));
        cache.slot_for_write("model", &prompt, 48);

        assert_eq!(cache.lookup("model", &prompt, 16, 32), None);
        assert_eq!(cache.lookup("model", &prompt, 32, 48), Some((48, 9)));
    }
}
