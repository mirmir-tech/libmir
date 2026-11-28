use std::sync::Arc;

use runtime::kv::BlockId;

use crate::engine::{
    Array, Error, KvCache, KvPageFormat, PagedArenaPool, PagedContextMode, Result, Stream,
};

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
fn keeps_fixed_arena_capacity_without_reallocating() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let stream = Stream::new_gpu_with_config_and_pool(Arc::default(), Arc::clone(&pool))?;
    let initial = Array::from_f32(&[1.0; 8], &[1, 1, 4, 2])?;
    let mut cache = KvCache::new_paged_with_pool_capacity(2, 1, KvPageFormat::Native, 12, pool, 0)?;
    drop(cache.update(&initial, &initial, &stream)?);
    let mut snapshot = cache.snapshot_at(4)?;
    snapshot.reserve(6)?;
    let extension = Array::from_f32(&[2.0; 4], &[1, 1, 2, 2])?;

    let context = snapshot.update(&extension, &extension, &stream)?;

    let pages = context.paged.ok_or(Error::NullHandle("paged context"))?.key_pages;
    assert_eq!(pages.shape()?[1], 12);
    Ok(())
}

#[test]
fn isolates_diverging_branches_via_runtime_block_ids() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let stream = Stream::new_gpu_with_config_and_pool(Arc::default(), Arc::clone(&pool))?;
    let prompt = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
    let mut owner = KvCache::new_paged_with_pool(2, 2, KvPageFormat::Native, Arc::clone(&pool), 0)?;
    owner.bind_blocks(&[BlockId(0), BlockId(1), BlockId(2)], None, &stream)?;
    let written = owner.update(&prompt, &prompt, &stream)?;
    written.keys.async_eval(&stream)?;
    stream.synchronize()?;

    let mut branch =
        KvCache::new_paged_with_pool(2, 2, KvPageFormat::Native, Arc::clone(&pool), 0)?;
    branch.bind_blocks(&[BlockId(0), BlockId(3), BlockId(4)], Some(2), &stream)?;

    let token = |value: f32| Array::from_f32(&[value, value], &[1, 1, 1, 2]);
    let fragmented = PagedContextMode::NativeIfFragmented;

    let owner_step =
        owner.update_for_attention_mode(&token(7.0)?, &token(7.0)?, &stream, 1, fragmented)?;
    let branch_step =
        branch.update_for_attention_mode(&token(9.0)?, &token(9.0)?, &stream, 1, fragmented)?;

    assert_eq!(owner_step.keys.to_vec_f32(&stream)?, [1.0, 2.0, 3.0, 4.0, 7.0, 7.0]);
    assert!(
        branch_step
            .paged
            .as_ref()
            .is_some_and(crate::engine::kv::PagedKvContext::is_fragmented)
    );
    let view = PagedContextMode::View;
    let owner_next =
        owner.update_for_attention_mode(&token(8.0)?, &token(8.0)?, &stream, 1, view)?;
    let branch_next =
        branch.update_for_attention_mode(&token(5.0)?, &token(5.0)?, &stream, 1, view)?;
    assert_eq!(owner_next.keys.to_vec_f32(&stream)?, [1.0, 2.0, 3.0, 4.0, 7.0, 7.0, 8.0, 8.0]);
    assert_eq!(branch_next.keys.to_vec_f32(&stream)?, [1.0, 2.0, 3.0, 4.0, 9.0, 9.0, 5.0, 5.0]);
    Ok(())
}
