use std::collections::{HashMap, VecDeque};

use super::{error::Result, session::SessionState};
use crate::engine::Array;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrefixKey([u8; 32]);

#[derive(Debug)]
struct PrefixSnapshot {
    state: SessionState,
    logits: Array,
}

#[derive(Debug)]
pub(super) struct PrefixCache {
    entries: HashMap<PrefixKey, PrefixSnapshot>,
    recency: VecDeque<PrefixKey>,
    capacity: usize,
}

impl PrefixCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            recency: VecDeque::new(),
            capacity,
        }
    }

    pub(super) fn restore_longest(
        &mut self,
        model: &str,
        tokens: &[u32],
    ) -> Result<Option<(SessionState, Array)>> {
        let Some(key) = prefix_keys(model, tokens)
            .into_iter()
            .rev()
            .find(|key| self.entries.contains_key(key))
        else {
            return Ok(None);
        };
        let (cache, logits, position) = {
            let Some(snapshot) = self.entries.get(&key) else {
                return Ok(None);
            };
            (
                snapshot.state.cache.snapshot_at(snapshot.state.position)?,
                snapshot.logits.snapshot()?,
                snapshot.state.position,
            )
        };
        self.touch(key);
        Ok(Some((SessionState::from_prefix(cache, position), logits)))
    }

    pub(super) fn insert(
        &mut self,
        model: &str,
        tokens: &[u32],
        state: &SessionState,
        logits: &Array,
    ) -> Result<()> {
        if self.capacity == 0 {
            return Ok(());
        }
        let key = prefix_key(model, tokens);
        let cache = state.cache.snapshot_at(state.position)?;
        let snapshot = PrefixSnapshot {
            state: SessionState::from_prefix(cache, state.position),
            logits: logits.snapshot()?,
        };
        let _previous = self.entries.insert(key, snapshot);
        self.touch(key);
        while self.entries.len() > self.capacity {
            if let Some(expired) = self.recency.pop_front() {
                let _removed = self.entries.remove(&expired);
            }
        }
        Ok(())
    }

    pub(super) const fn enabled(&self) -> bool {
        self.capacity > 0
    }

    pub(super) const fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
    }

    fn touch(&mut self, key: PrefixKey) {
        if let Some(index) = self.recency.iter().position(|entry| *entry == key) {
            let _removed = self.recency.remove(index);
        }
        self.recency.push_back(key);
    }
}

fn prefix_key(model: &str, tokens: &[u32]) -> PrefixKey {
    let mut hasher = prefix_hasher(model);
    for token in tokens {
        hasher.update(&token.to_le_bytes());
    }
    PrefixKey(*hasher.finalize().as_bytes())
}

fn prefix_keys(model: &str, tokens: &[u32]) -> Vec<PrefixKey> {
    let mut hasher = prefix_hasher(model);
    let mut keys = Vec::with_capacity(tokens.len());
    for token in tokens {
        hasher.update(&token.to_le_bytes());
        keys.push(PrefixKey(*hasher.finalize().as_bytes()));
    }
    keys
}

fn prefix_hasher(model: &str) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model.as_bytes());
    hasher.update(&[0]);
    hasher
}
