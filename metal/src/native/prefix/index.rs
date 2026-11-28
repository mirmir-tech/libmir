use std::collections::HashMap;

use foundation::model::ModelId;

use super::PrefixEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PrefixKey(pub(super) [u8; 32]);

/// The full prompt, plus block boundaries in its second half that the
/// snapshot can rewind to (`floor`,
/// [`crate::engine::DecoderCache::prefix_floor`]).
pub(super) fn indexed_prefixes(
    model: &ModelId,
    tokens: &[u32],
    block_size: Option<usize>,
    floor: usize,
) -> Vec<(PrefixKey, usize)> {
    let mut hasher = prefix_hasher(model);
    let mut indexed = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        hasher.update(&token.to_le_bytes());
        let position = index + 1;
        let efficient = position.saturating_mul(2) >= tokens.len();
        if position == tokens.len()
            || efficient && position >= floor && block_size.is_some_and(|size| position % size == 0)
        {
            indexed.push((PrefixKey(*hasher.finalize().as_bytes()), position));
        }
    }
    indexed
}

impl PrefixEntry {
    /// Continuations restore `position` rounded down to a block, and complete
    /// prompts without logits the block before their last token, wherever the
    /// snapshot rewinds that far; otherwise the exact positions.
    pub(super) fn new(
        memory_group: u64,
        position: usize,
        block_size: Option<usize>,
        floor: usize,
    ) -> Self {
        let aligned = |target: usize| {
            block_size.map_or(target, |size| {
                let rounded = target / size * size;
                if rounded >= floor {
                    rounded
                } else {
                    target
                }
            })
        };
        Self {
            memory_group,
            position,
            continuation_position: aligned(position),
            completion_position: aligned(position.saturating_sub(1)),
        }
    }
}

pub(super) fn longest_indexed_prefix(
    model: &ModelId,
    tokens: &[u32],
    entries: &HashMap<PrefixKey, PrefixEntry>,
) -> Option<(PrefixKey, PrefixEntry)> {
    longest_usable_prefix(model, tokens, entries, |_| true)
}

pub(super) fn longest_usable_prefix(
    model: &ModelId,
    tokens: &[u32],
    entries: &HashMap<PrefixKey, PrefixEntry>,
    usable: impl Fn(&PrefixEntry) -> bool,
) -> Option<(PrefixKey, PrefixEntry)> {
    let mut hasher = prefix_hasher(model);
    let mut longest = None;
    for token in tokens {
        hasher.update(&token.to_le_bytes());
        let key = PrefixKey(*hasher.finalize().as_bytes());
        if let Some(entry) = entries.get(&key).filter(|entry| usable(entry)) {
            longest = Some((key, *entry));
        }
    }
    longest
}

fn prefix_hasher(model: &ModelId) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model.as_str().as_bytes());
    hasher.update(&[0]);
    hasher
}
