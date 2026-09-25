use std::sync::Arc;

use crate::{RuntimeError, error::Result};

/// Immutable request-owned token allowlist. Padding bits are always cleared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMask {
    words: Arc<[u32]>,
    vocab: usize,
}

impl TokenMask {
    pub fn new(mut words: Vec<u32>, vocab: usize) -> Result<Self> {
        if vocab == 0 || words.len() != vocab.div_ceil(32) {
            return Err(RuntimeError::Backend("token mask vocabulary mismatch".into()));
        }
        if !vocab.is_multiple_of(32) {
            let last = words.len() - 1;
            words[last] &= (1_u32 << (vocab % 32)) - 1;
        }
        if words.iter().all(|word| *word == 0) {
            return Err(RuntimeError::Backend("token mask rejects every token".into()));
        }
        Ok(Self { words: words.into(), vocab })
    }

    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    #[must_use]
    pub const fn vocab(&self) -> usize {
        self.vocab
    }
}

/// Sampling over a masked distribution; host-history policies are excluded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceSampling {
    Greedy,
    Random {
        temperature: f32,
        top_p: f32,
        top_k: usize,
        draw: f32,
    },
}
