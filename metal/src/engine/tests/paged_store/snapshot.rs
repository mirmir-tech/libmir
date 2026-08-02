use crate::engine::{Array, Error, KvCache, Result, Stream};

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
fn grows_a_snapshot_to_its_reserved_capacity_without_doubling() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let initial = Array::from_f32(&[1.0; 8], &[1, 1, 4, 2])?;
    let mut cache = KvCache::new_paged(2, 1)?;
    drop(cache.update(&initial, &initial, &stream)?);
    let mut snapshot = cache.snapshot_at(4)?;
    snapshot.reserve(6)?;
    let extension = Array::from_f32(&[2.0; 4], &[1, 1, 2, 2])?;

    let context = snapshot.update(&extension, &extension, &stream)?;

    let pages = context.paged.ok_or(Error::NullHandle("paged context"))?.key_pages;
    assert_eq!(pages.shape()?[1], 6);
    Ok(())
}
