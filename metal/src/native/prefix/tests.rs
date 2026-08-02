use super::{
    PrefixCache, PrefixEntry, PrefixSnapshot,
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
    cache.entries.insert(key, PrefixEntry { memory_group, position: 1 });
    cache.groups.entry(memory_group).or_insert(PrefixSnapshot {
        state: SessionState::new(DecoderCache::new(&[], 1)?),
        logits: Array::from_u32(&[0], &[1])?,
        bytes,
    });
    cache.touch_group(memory_group);
    Ok(())
}
