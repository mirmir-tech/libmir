use std::collections::HashMap;

use super::{BlockHash, BlockId};

#[derive(Debug, Clone)]
pub struct PrefixProbe {
    pub cached_blocks: Vec<BlockId>,
    pub cached_tokens: usize,
    pub missing_tokens: usize,
    pub last_hash: Option<BlockHash>,
}

#[derive(Debug)]
struct Entry {
    block: BlockId,
    older: Option<BlockHash>,
    newer: Option<BlockHash>,
}

#[derive(Debug, Default)]
pub struct PrefixCache {
    blocks: HashMap<BlockHash, Entry>,
    oldest: Option<BlockHash>,
    newest: Option<BlockHash>,
}

impl PrefixCache {
    pub fn insert(&mut self, hash: BlockHash, block: BlockId) -> Option<BlockId> {
        if let Some(previous) = self
            .blocks
            .get_mut(&hash)
            .map(|entry| std::mem::replace(&mut entry.block, block))
        {
            self.touch(hash);
            return Some(previous);
        }
        let older = self.newest;
        self.blocks.insert(hash, Entry { block, older, newer: None });
        if let Some(previous) = older {
            if let Some(entry) = self.blocks.get_mut(&previous) {
                entry.newer = Some(hash);
            }
        } else {
            self.oldest = Some(hash);
        }
        self.newest = Some(hash);
        None
    }

    #[must_use]
    pub fn peek(&self, hash: BlockHash) -> Option<BlockId> {
        self.blocks.get(&hash).map(|entry| entry.block)
    }

    pub fn touch(&mut self, hash: BlockHash) {
        if self.newest == Some(hash) || !self.blocks.contains_key(&hash) {
            return;
        }
        self.detach(hash);
        let previous = self.newest;
        if let Some(entry) = self.blocks.get_mut(&hash) {
            entry.older = previous;
            entry.newer = None;
        }
        if let Some(previous) = previous {
            if let Some(entry) = self.blocks.get_mut(&previous) {
                entry.newer = Some(hash);
            }
        } else {
            self.oldest = Some(hash);
        }
        self.newest = Some(hash);
    }

    pub fn remove(&mut self, hash: [u8; 32]) -> Option<BlockId> {
        let hash = BlockHash(hash);
        self.detach(hash);
        self.blocks.remove(&hash).map(|entry| entry.block)
    }

    #[must_use]
    pub fn oldest_first(&self) -> Vec<BlockHash> {
        let mut hashes = Vec::with_capacity(self.blocks.len());
        let mut current = self.oldest;
        while let Some(hash) = current {
            hashes.push(hash);
            current = self.blocks.get(&hash).and_then(|entry| entry.newer);
        }
        hashes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    fn detach(&mut self, hash: BlockHash) {
        let Some((older, newer)) = self.blocks.get(&hash).map(|entry| (entry.older, entry.newer))
        else {
            return;
        };
        if let Some(older) = older {
            if let Some(entry) = self.blocks.get_mut(&older) {
                entry.newer = newer;
            }
        } else {
            self.oldest = newer;
        }
        if let Some(newer) = newer {
            if let Some(entry) = self.blocks.get_mut(&newer) {
                entry.older = older;
            }
        } else {
            self.newest = older;
        }
    }
}

/// Prefix tokens a backend may skip during prefill given `cached_tokens`
/// resident in the runtime block table.
///
/// When `replay_tokens > 0` (e.g. sliding-window replay), backs off by that
/// margin. When the entire prompt is already cached, backs off by one block so
/// the final token is always evaluated to produce fresh logits.
#[must_use]
pub fn reusable_prefix_tokens(
    cached_tokens: usize,
    prompt_tokens: usize,
    block_size: Option<usize>,
    replay_tokens: usize,
) -> usize {
    let cached_tokens = cached_tokens.min(prompt_tokens);
    if replay_tokens > 0 {
        return cached_tokens.saturating_sub(replay_tokens);
    }
    if cached_tokens < prompt_tokens {
        return cached_tokens;
    }
    block_size
        .filter(|size| *size > 0)
        .map_or(0, |size| cached_tokens.saturating_sub(size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_and_remove_preserve_constant_time_recency_links() {
        let hashes = [1_u8, 2, 3].map(|value| BlockHash([value; 32]));
        let mut cache = PrefixCache::default();
        for (hash, block) in hashes.into_iter().zip([BlockId(0), BlockId(1), BlockId(2)]) {
            assert_eq!(cache.insert(hash, block), None);
        }
        cache.touch(hashes[0]);
        assert_eq!(cache.oldest_first(), [hashes[1], hashes[2], hashes[0]]);
        assert_eq!(cache.remove(hashes[2].0), Some(BlockId(2)));
        assert_eq!(cache.oldest_first(), [hashes[1], hashes[0]]);
    }

    #[test]
    fn backs_off_exact_prompt_hits_by_one_block_and_respects_replay_margin() {
        assert_eq!(reusable_prefix_tokens(32, 48, Some(16), 0), 32);
        assert_eq!(reusable_prefix_tokens(32, 32, Some(16), 0), 16);
        assert_eq!(reusable_prefix_tokens(32, 32, Some(16), 8), 24);
        assert_eq!(reusable_prefix_tokens(32, 32, None, 0), 0);
    }
}
