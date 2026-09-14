use super::*;
use crate::engine::PagedContextMode;

fn append(cache: &mut KvCache, count: usize, value: f32, stream: &Stream) -> Result<Vec<f32>> {
    let input = Array::from_f32(&vec![value; count * 4], &[1, 1, i32::try_from(count)?, 4])?;
    let context =
        cache.update_for_attention_mode(&input, &input, stream, 0, PagedContextMode::Both)?;
    context.keys.to_vec_f32(stream)
}

#[test]
fn reserved_decode_tail_preserves_cow_and_releases_unused_pages() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pool = Arc::new(PagedArenaPool::default());
    let make = || {
        KvCache::new_paged_with_pool_capacity(4, 4, KvPageFormat::Native, 12, Arc::clone(&pool), 0)
    };
    let available = || pool.available_pages(12, 0, 1, 4, KvPageFormat::Native);
    let mut cohort = Vec::new();
    for _ in 0..3 {
        let mut cache = make()?;
        cache.plan_contiguous(12);
        assert_eq!(append(&mut cache, 3, 0.5, &stream)?, vec![0.5; 12]);
        cohort.push(cache);
    }
    assert_eq!(available()?, 3); // Includes six reserved but not yet written pages.
    let mut fork = cohort[0].snapshot_at(3)?;
    fork.plan_decode_capacity(1, 0, &mut DecodeCapacity::default())?;
    assert_eq!(append(&mut fork, 1, 0.75, &stream)?, [vec![0.5; 12], vec![0.75; 4]].concat());
    assert_eq!(available()?, 0);
    assert_ne!(fork.first_physical_page(), cohort[0].first_physical_page());
    assert_eq!(
        append(&mut cohort[0], 1, 0.25, &stream)?,
        [vec![0.5; 12], vec![0.25; 4]].concat()
    );
    fork.reset()?;
    assert_eq!(available()?, 3);
    cohort[1].reset()?;
    assert_eq!(available()?, 6);
    let mut refill = make()?;
    refill.plan_contiguous(12);
    assert_eq!(append(&mut refill, 3, 0.125, &stream)?, vec![0.125; 12]);
    for cache in [&mut cohort[0], &mut refill] {
        let remaining = 12 - cache.offset()?;
        cache.plan_decode_capacity(remaining, 0, &mut DecodeCapacity::default())?;
        let _values = append(cache, remaining, 0.0, &stream)?;
    }
    assert_eq!(available()?, 3);
    drop(cohort);
    drop(refill);
    drop(fork);
    assert_eq!(available()?, 12);
    assert_eq!(pool.resident_arenas()?, 0);
    Ok(())
}

#[test]
fn enlarged_reservation_is_rejected_before_decode_mutates_cache() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = cache(4, 4, 3, KvPageFormat::Native)?;
    cache.plan_contiguous(4);
    let _values = append(&mut cache, 3, 0.5, &stream)?;
    cache.plan_contiguous(16);
    assert!(cache.plan_decode_capacity(1, 0, &mut DecodeCapacity::default()).is_err());
    assert_eq!(cache.offset()?, 3);
    assert_eq!(cache.physical_page_count(), 1);
    cache.plan_contiguous(12);
    cache.plan_decode_capacity(1, 0, &mut DecodeCapacity::default())?;
    assert_eq!(append(&mut cache, 1, 0.25, &stream)?, [vec![0.5; 12], vec![0.25; 4]].concat());
    Ok(())
}

#[test]
fn reclaiming_unwritten_tail_preserves_shared_history_and_cow() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pool = Arc::new(PagedArenaPool::default());
    let make = || {
        KvCache::new_paged_with_pool_capacity(4, 4, KvPageFormat::Native, 12, Arc::clone(&pool), 0)
    };
    let available = || pool.available_pages(12, 0, 1, 4, KvPageFormat::Native);
    let mut original = make()?;
    original.plan_contiguous(32);
    assert_eq!(append(&mut original, 3, 0.5, &stream)?, vec![0.5; 12]);
    let mut fork = original.snapshot_at(3)?;
    original.release_reservation_after(7)?;
    assert_eq!(available()?, 10);
    let mut refill = make()?;
    refill.plan_contiguous(32);
    let _values = append(&mut refill, 4, 0.125, &stream)?;
    fork.plan_contiguous(4);
    fork.plan_decode_capacity(1, 0, &mut DecodeCapacity::default())?;
    assert_eq!(append(&mut fork, 1, 0.75, &stream)?, [vec![0.5; 12], vec![0.75; 4]].concat());
    original.plan_decode_capacity(1, 0, &mut DecodeCapacity::default())?;
    assert_eq!(
        append(&mut original, 1, 0.25, &stream)?,
        [vec![0.5; 12], vec![0.25; 4]].concat()
    );
    assert_eq!(original.physical_page_count(), 1);
    assert_ne!(fork.first_physical_page(), original.first_physical_page());
    drop((original, fork, refill));
    assert_eq!(available()?, 12);
    assert_eq!(pool.resident_arenas()?, 0);
    Ok(())
}
