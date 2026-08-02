use std::collections::HashMap;

use super::PrefixEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PrefixKey(pub(super) [u8; 32]);

pub(super) fn indexed_prefixes(
    model: &str,
    tokens: &[u32],
    block_size: Option<usize>,
) -> Vec<(PrefixKey, usize)> {
    let mut hasher = prefix_hasher(model);
    let mut indexed = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        hasher.update(&token.to_le_bytes());
        let position = index + 1;
        let efficient = position.saturating_mul(2) >= tokens.len();
        if position == tokens.len()
            || efficient && block_size.is_some_and(|size| position % size == 0)
        {
            indexed.push((PrefixKey(*hasher.finalize().as_bytes()), position));
        }
    }
    indexed
}

pub(super) fn longest_indexed_prefix(
    model: &str,
    tokens: &[u32],
    entries: &HashMap<PrefixKey, PrefixEntry>,
) -> Option<(PrefixKey, PrefixEntry)> {
    let mut hasher = prefix_hasher(model);
    let mut longest = None;
    for token in tokens {
        hasher.update(&token.to_le_bytes());
        let key = PrefixKey(*hasher.finalize().as_bytes());
        if let Some(entry) = entries.get(&key) {
            longest = Some((key, *entry));
        }
    }
    longest
}

fn prefix_hasher(model: &str) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model.as_bytes());
    hasher.update(&[0]);
    hasher
}
