#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub [u8; 32]);

#[derive(Debug, Clone)]
pub struct KvBlock {
    pub id: BlockId,
    pub ref_count: u32,
    pub hash: Option<[u8; 32]>,
    pub token_count: usize,
    is_free: bool,
}

impl KvBlock {
    #[must_use]
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            ref_count: 0,
            hash: None,
            token_count: 0,
            is_free: true,
        }
    }

    pub fn allocate(&mut self) {
        self.is_free = false;
        self.ref_count = 1;
    }

    pub fn reset(&mut self) {
        self.ref_count = 0;
        self.hash = None;
        self.token_count = 0;
        self.is_free = true;
    }

    #[must_use]
    pub fn is_free(&self) -> bool {
        self.is_free
    }
}

impl BlockHash {
    #[must_use]
    pub fn from_tokens(model: &str, parent: Option<Self>, tokens: &[u32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model.as_bytes());
        if let Some(parent) = parent {
            hasher.update(&parent.0);
        }
        for token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }
}
