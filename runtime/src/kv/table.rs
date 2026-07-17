use super::BlockId;

#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    block_size: usize,
    token_len: usize,
}

impl BlockTable {
    #[must_use]
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size,
            token_len: 0,
        }
    }

    pub fn push(&mut self, block: BlockId) {
        self.blocks.push(block);
    }

    pub fn set_token_len(&mut self, token_len: usize) {
        self.token_len = token_len;
    }

    #[must_use]
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    #[must_use]
    pub fn token_len(&self) -> usize {
        self.token_len
    }

    #[must_use]
    pub fn block_size(&self) -> Option<usize> {
        (self.block_size > 0).then_some(self.block_size)
    }

    #[must_use]
    pub fn capacity(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
