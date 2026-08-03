use std::collections::HashMap;

use super::{
    PrefixCache, PrefixEntry, PrefixGroup, PrefixSnapshot,
    index::{PrefixKey, indexed_prefixes},
};
use crate::{
    engine::{Array, DecoderCache},
    native::{error::Result, session::SessionState},
};

#[test]
fn indexes_storage_efficient_complete_blocks_and_the_exact_prompt() {
    let indexed = indexed_prefixes("model", &[1, 2, 3, 4, 5], Some(2));
    assert_eq!(indexed.iter().map(|(_, position)| *position).collect::<Vec<_>>(), [4, 5]);
}

#[test]
fn indexes_only_the_exact_prompt_without_block_prefixes() {
    let indexed = indexed_prefixes("model", &[1, 2, 3], None);
    assert_eq!(indexed.iter().map(|(_, position)| *position).collect::<Vec<_>>(), [3]);
}

#[test]
fn continuation_replays_an_unaligned_terminal_page() -> Result<()> {
    let mut cache = PrefixCache::new(2, usize::MAX);
    let mut state = SessionState::new(DecoderCache::new(&[], 1)?);
    state.position = 5;
    let logits = Array::from_u32(&[0], &[1])?;
    cache.insert("model", &[1, 2, 3, 4, 5], &state, &logits, Some(2), 100)?;

    let (continued, continued_logits) = cache
        .restore_longest("model", &[1, 2, 3, 4, 5, 6])?
        .ok_or(crate::native::error::Error::NoPrefixLogits)?;
    let (exact, exact_logits) = cache
        .restore_longest("model", &[1, 2, 3, 4, 5])?
        .ok_or(crate::native::error::Error::NoPrefixLogits)?;

    assert_eq!(continued.position, 4);
    assert!(continued_logits.is_none());
    assert_eq!(exact.position, 5);
    assert!(exact_logits.is_some());
    Ok(())
}

#[test]
fn continuation_replays_an_unaligned_checkpoint_page() -> Result<()> {
    let mut cache = PrefixCache::new(2, usize::MAX);
    let mut state = SessionState::new(DecoderCache::new(&[], 1)?);
    state.position = 5;
    cache.insert_checkpoint("model", &[1, 2, 3, 4, 5], &state, 2, 100)?;

    let (continued, logits) = cache
        .restore_longest("model", &[1, 2, 3, 4, 5, 6])?
        .ok_or(crate::native::error::Error::NoPrefixLogits)?;

    assert_eq!(continued.position, 4);
    assert!(logits.is_none());

    let (completed, completed_logits) = cache
        .restore_longest("model", &[1, 2, 3, 4, 5])?
        .ok_or(crate::native::error::Error::NoPrefixLogits)?;
    assert_eq!(completed.position, 4);
    assert!(completed_logits.is_none());
    Ok(())
}

#[test]
fn message_checkpoint_shares_a_group_with_the_completed_prompt() -> Result<()> {
    let mut cache = PrefixCache::new(2, usize::MAX);
    let mut state = SessionState::new(DecoderCache::new(&[], 1)?);
    state.position = 2;
    cache.insert_checkpoint("model", &[1, 2], &state, 2, 40)?;
    state.position = 3;
    let logits = Array::from_u32(&[0], &[1])?;
    cache.insert("model", &[1, 2, 3], &state, &logits, None, 60)?;

    let (restored, logits) = cache
        .restore_longest("model", &[1, 2, 9])?
        .ok_or(crate::native::error::Error::NoPrefixLogits)?;

    assert_eq!(restored.position, 2);
    assert!(logits.is_none());
    assert_eq!(cache.groups.len(), 1);
    assert_eq!(cache.resident_bytes(), 100);
    Ok(())
}

#[test]
fn counts_each_snapshot_group_once() -> Result<()> {
    let mut cache = PrefixCache::new(2, usize::MAX);
    insert_snapshot(&mut cache, 1, 1, 100)?;
    insert_snapshot(&mut cache, 2, 1, 100)?;
    assert_eq!(cache.resident_bytes(), 100);
    Ok(())
}

#[test]
fn evicts_complete_sequence_groups_in_lru_order() -> Result<()> {
    let mut cache = PrefixCache::new(2, usize::MAX);
    insert_snapshot(&mut cache, 1, 1, 100)?;
    insert_snapshot(&mut cache, 2, 2, 100)?;
    cache.touch_group(1);
    insert_snapshot(&mut cache, 3, 3, 100)?;
    cache.enforce_limits();

    assert!(!cache.groups.contains_key(&2));
    assert!(cache.entries.values().all(|entry| entry.memory_group != 2));
    assert_eq!(cache.group_recency.iter().copied().collect::<Vec<_>>(), [1, 3]);
    Ok(())
}

#[test]
fn reserves_one_group_before_computing_a_cache_miss() -> Result<()> {
    let mut cache = PrefixCache::new(2, usize::MAX);
    insert_snapshot(&mut cache, 1, 1, 100)?;
    insert_snapshot(&mut cache, 2, 2, 100)?;

    assert!(cache.restore_longest("missing", &[1])?.is_none());

    assert!(!cache.groups.contains_key(&1));
    assert!(cache.groups.contains_key(&2));
    Ok(())
}

#[test]
fn batch_admission_reserves_slots_after_prefix_states_are_restored() -> Result<()> {
    let mut cache = PrefixCache::new(3, usize::MAX);
    insert_snapshot(&mut cache, 1, 1, 100)?;
    insert_snapshot(&mut cache, 2, 2, 100)?;
    insert_snapshot(&mut cache, 3, 3, 100)?;

    assert!(cache.reserve_batch_slots(2));

    assert!(!cache.groups.contains_key(&1));
    assert!(!cache.groups.contains_key(&2));
    assert!(cache.groups.contains_key(&3));
    Ok(())
}

#[test]
fn batch_admission_uses_available_capacity_before_evicting() -> Result<()> {
    let mut cache = PrefixCache::new(3, usize::MAX);
    insert_snapshot(&mut cache, 1, 1, 100)?;

    assert!(!cache.reserve_batch_slots(2));
    assert!(cache.groups.contains_key(&1));
    Ok(())
}

fn insert_snapshot(
    cache: &mut PrefixCache,
    key: u8,
    memory_group: u64,
    bytes: usize,
) -> Result<()> {
    insert_snapshot_with_key(cache, PrefixKey([key; 32]), memory_group, bytes)
}

fn insert_snapshot_with_key(
    cache: &mut PrefixCache,
    key: PrefixKey,
    memory_group: u64,
    bytes: usize,
) -> Result<()> {
    cache.entries.insert(
        key,
        PrefixEntry {
            memory_group,
            position: 1,
            continuation_position: 1,
            completion_position: 0,
        },
    );
    cache.groups.entry(memory_group).or_insert(PrefixGroup {
        terminal: Some(PrefixSnapshot {
            state: SessionState::new(DecoderCache::new(&[], 1)?),
            logits: Some(Array::from_u32(&[0], &[1])?),
        }),
        checkpoints: HashMap::new(),
        bytes,
    });
    cache.touch_group(memory_group);
    Ok(())
}
