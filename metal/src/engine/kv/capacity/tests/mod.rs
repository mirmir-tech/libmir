use std::sync::Arc;
mod reservation;

use super::*;
use crate::engine::{Array, DecoderCache, KvPageFormat, PagedArenaPool, Stream};

fn cache(step: usize, page_size: usize, maximum: usize, format: KvPageFormat) -> Result<KvCache> {
    KvCache::new_paged_with_pool_capacity(
        step,
        page_size,
        format,
        maximum,
        Arc::new(PagedArenaPool::default()),
        0,
    )
}

fn tokens(count: usize) -> Result<Array> {
    Array::from_f32(&vec![0.5; count * 4], &[1, 1, i32::try_from(count)?, 4])
}

fn write(cache: &mut KvCache, count: usize, stream: &Stream) -> Result<()> {
    let input = tokens(count)?;
    let context = cache.update(&input, &input, stream)?;
    context
        .paged
        .ok_or(Error::NullHandle("pages"))?
        .page_dependency
        .async_eval(stream)?;
    stream.synchronize()
}

#[test]
fn last_shared_writer_reuses_the_original_page() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for format in [KvPageFormat::Native, KvPageFormat::Int8PerTokenHead] {
        let mut first = cache(4, 4, 2, format)?;
        write(&mut first, 1, &stream)?;
        let mut second = first.snapshot_at(1)?;
        let mut plan = DecodeCapacity::default();
        first.plan_decode_capacity(1, 1, &mut plan)?;
        second.plan_decode_capacity(1, 1, &mut plan)?;
        assert_eq!(first.first_physical_page(), second.first_physical_page());
        write(&mut first, 1, &stream)?;
        write(&mut second, 1, &stream)?;
        assert_ne!(first.first_physical_page(), second.first_physical_page());
    }
    Ok(())
}

#[test]
fn reserved_pages_cover_two_forwards_without_free_capacity() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = cache(8, 4, 2, KvPageFormat::Native)?;
    write(&mut cache, 3, &stream)?;
    cache.plan_decode_capacity(2, 1, &mut DecodeCapacity::default())?;
    write(&mut cache, 1, &stream)?;
    write(&mut cache, 1, &stream)?;
    assert_eq!(cache.offset()?, 5);
    Ok(())
}

#[test]
fn promotion_budget_includes_both_forwards() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = cache(2, 2, 1, KvPageFormat::Native)?;
    let token = tokens(1)?;
    drop(cache.update_with_page_min_context(&token, &token, &stream, 3)?);
    cache.plan_decode_capacity(1, 3, &mut DecodeCapacity::default())?;
    assert!(cache.plan_decode_capacity(2, 3, &mut DecodeCapacity::default()).is_err());
    assert_eq!(cache.offset()?, 1);
    assert_eq!(cache.physical_page_count(), 0);
    Ok(())
}

#[test]
fn arena_growth_does_not_round_past_its_physical_limit() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for format in [KvPageFormat::Native, KvPageFormat::Int8PerTokenHead] {
        let mut cache = cache(4, 2, 3, format)?;
        write(&mut cache, 4, &stream)?;
        write(&mut cache, 1, &stream)?;
        cache.plan_decode_capacity(1, 1, &mut DecodeCapacity::default())?;
        write(&mut cache, 1, &stream)?;
        assert_eq!(cache.offset()?, 6);
        assert!(cache.plan_decode_capacity(1, 1, &mut DecodeCapacity::default()).is_err());
    }
    Ok(())
}

#[test]
fn later_layer_capacity_is_checked_before_any_cache_changes() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pool = Arc::new(PagedArenaPool::default());
    let mut cache =
        DecoderCache::new_with_pool_capacity(&[None, None], 4, KvPageFormat::Native, 4, 2, &pool)?;
    for layer in cache.attention_caches_mut()? {
        write(layer, 4, &stream)?;
    }
    let mut blocker =
        KvCache::new_paged_with_pool_capacity(4, 4, KvPageFormat::Native, 2, pool, 1)?;
    write(&mut blocker, 4, &stream)?;
    assert!(cache.plan_decode_capacity(1, 1, &mut DecodeCapacity::default()).is_err());
    for layer in cache.attention_caches_mut()? {
        assert_eq!(layer.offset()?, 4);
    }
    blocker.reset()?;
    cache.plan_decode_capacity(1, 1, &mut DecodeCapacity::default())?;
    for layer in cache.attention_caches_mut()? {
        write(layer, 1, &stream)?;
    }
    assert_eq!(cache.cached_tokens()?, 5);
    Ok(())
}
