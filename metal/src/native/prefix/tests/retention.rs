use super::*;

#[test]
fn leased_prefixes_survive_eviction_between_physical_waves() -> Result<()> {
    let mut cache = PrefixCache::new(3, usize::MAX);
    insert_prompt(&mut cache, 1)?;
    insert_prompt(&mut cache, 2)?;
    insert_prompt(&mut cache, 3)?;

    let leased = [1, 2, 3]
        .into_iter()
        .map(|token| cache.lease_longest("model", &[token]))
        .collect::<Result<Vec<_>>>()?;
    assert!(cache.reserve_batch_slots(2));
    assert_eq!(cache.groups.len(), 1);
    assert!(leased.into_iter().all(|prefix| {
        prefix.is_some_and(|prefix| prefix.restored.0.position == 1 && prefix.restored.1.is_some())
    }));
    Ok(())
}

#[test]
fn logical_cohort_releases_only_leased_source_groups() -> Result<()> {
    let mut cache = PrefixCache::new(3, usize::MAX);
    insert_prompt(&mut cache, 1)?;
    insert_prompt(&mut cache, 2)?;
    insert_prompt(&mut cache, 3)?;

    let leased = [1, 2]
        .into_iter()
        .map(|token| cache.lease_longest("model", &[token]))
        .collect::<Result<Vec<_>>>()?;
    let groups = leased
        .iter()
        .filter_map(|prefix| prefix.as_ref().map(|prefix| prefix.memory_group))
        .collect();
    assert!(cache.evict_groups(&groups));

    assert_eq!(cache.groups.len(), 1);
    assert!(cache.restore_longest("model", &[3])?.is_some());
    assert!(leased.into_iter().all(|prefix| prefix.is_some()));
    Ok(())
}
