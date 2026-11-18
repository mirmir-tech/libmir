use crate::engine::{Array, Error, KvCache, PagedContextMode, Result, Stream};

#[test]
fn retains_only_pages_reachable_from_the_snapshot_offset() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let values = Array::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 3, 2])?;
    let mut cache = KvCache::new_paged(2, 2)?;
    drop(cache.update(&values, &values, &stream)?);

    let snapshot = cache.snapshot_at(1)?;

    assert_eq!(cache.physical_page_count(), 2);
    assert_eq!(snapshot.physical_page_count(), 1);
    Ok(())
}

#[test]
fn grows_a_snapshot_with_amortized_capacity() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let initial = Array::from_f32(&[1.0; 8], &[1, 1, 4, 2])?;
    let mut cache = KvCache::new_paged(2, 1)?;
    drop(cache.update(&initial, &initial, &stream)?);
    let mut snapshot = cache.snapshot_at(4)?;
    snapshot.reserve(6)?;
    let extension = Array::from_f32(&[2.0; 4], &[1, 1, 2, 2])?;

    let context = snapshot.update(&extension, &extension, &stream)?;

    let pages = context.paged.ok_or(Error::NullHandle("paged context"))?.key_pages;
    assert_eq!(pages.shape()?[1], 8);
    Ok(())
}

#[test]
fn keeps_the_owner_contiguous_after_a_snapshot_of_a_partial_page() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let prompt = Array::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 3, 2])?;
    let mut owner = KvCache::new_paged(2, 2)?;
    drop(owner.update(&prompt, &prompt, &stream)?);
    let snapshot = owner.snapshot_at(3)?;
    let mut restored = snapshot.snapshot_at(3)?;
    let token = |value: f32| Array::from_f32(&[value, value], &[1, 1, 1, 2]);
    let fragmented = PagedContextMode::NativeIfFragmented;

    let owner_step =
        owner.update_for_attention_mode(&token(7.0)?, &token(7.0)?, &stream, 1, fragmented)?;
    let restored_step =
        restored.update_for_attention_mode(&token(9.0)?, &token(9.0)?, &stream, 1, fragmented)?;

    assert!(owner_step.paged.is_none());
    assert_eq!(owner_step.keys.to_vec_f32(&stream)?, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0]);
    assert!(restored_step.paged.as_ref().is_some_and(|paged| paged.fragmented));
    let view = PagedContextMode::View;
    let owner_next =
        owner.update_for_attention_mode(&token(8.0)?, &token(8.0)?, &stream, 1, view)?;
    let restored_next =
        restored.update_for_attention_mode(&token(5.0)?, &token(5.0)?, &stream, 1, view)?;
    assert_eq!(
        owner_next.keys.to_vec_f32(&stream)?,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0]
    );
    assert_eq!(
        restored_next.keys.to_vec_f32(&stream)?,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 9.0, 5.0, 5.0]
    );
    Ok(())
}
