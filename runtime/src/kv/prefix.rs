use std::collections::HashMap;

use super::{BlockHash, BlockId};

#[derive(Debug, Clone)]
pub struct PrefixProbe {
    pub cached_blocks: Vec<BlockId>,
    pub cached_tokens: usize,
    pub missing_tokens: usize,
    pub last_hash: Option<BlockHash>,
}

#[derive(Debug, Default)]
pub struct PrefixCache {
    blocks: HashMap<BlockHash, BlockId>,
}

impl PrefixCache {
    pub fn insert(&mut self, hash: BlockHash, block: BlockId) {
        let _previous = self.blocks.insert(hash, block);
    }

    #[must_use]
    pub fn get(&self, hash: BlockHash) -> Option<BlockId> {
        self.blocks.get(&hash).copied()
    }

    pub fn remove(&mut self, hash: [u8; 32]) -> Option<BlockId> {
        self.blocks.remove(&BlockHash(hash))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
