use std::sync::Arc;

use crate::engine::{
    Array, KvCache, KvPageFormat, PagedArenaPool, PagedContextMode, Result, Stream,
};

#[test]
fn isolates_sessions_while_sharing_a_physical_arena() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let first_stream = stream(&pool)?;
    let second_stream = stream(&pool)?;
    let mut first = cache(&pool, 0)?;
    let mut second = cache(&pool, 0)?;
    let first_value = values(&[1.0, 2.0])?;
    let second_value = values(&[3.0, 4.0])?;

    let first_context = first.update(&first_value, &first_value, &first_stream)?;
    first_context.keys.async_eval(&first_stream)?;
    first_stream.synchronize()?;
    let second_context = second.update(&second_value, &second_value, &second_stream)?;
    second_context.keys.async_eval(&second_stream)?;
    second_stream.synchronize()?;

    assert!(first.shares_paged_arena(&second));
    assert_ne!(first.first_physical_page(), second.first_physical_page());
    assert_eq!(first_context.keys.to_vec_f32(&first_stream)?, vec![1.0, 2.0]);
    assert_eq!(second_context.keys.to_vec_f32(&second_stream)?, vec![3.0, 4.0]);
    assert_eq!(pool.resident_arenas()?, 1);
    Ok(())
}

#[test]
fn recycles_released_pages_across_sessions() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let stream = stream(&pool)?;
    let mut first = cache(&pool, 0)?;
    let mut second = cache(&pool, 0)?;
    let value = values(&[1.0, 2.0])?;
    drop(first.update(&value, &value, &stream)?);
    drop(second.update(&value, &value, &stream)?);
    let released = first.first_physical_page();
    first.reset()?;

    let mut replacement = cache(&pool, 0)?;
    drop(replacement.update(&value, &value, &stream)?);

    assert_eq!(replacement.first_physical_page(), released);
    assert!(replacement.shares_paged_arena(&second));
    Ok(())
}

#[test]
fn gathers_multiple_physical_runs_in_logical_order() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let stream = stream(&pool)?;
    let mut first = cache(&pool, 0)?;
    let mut blocker = cache(&pool, 0)?;
    let initial = values(&[1.0, 2.0])?;
    let occupied = values(&[9.0, 9.0])?;
    drop(first.update(&initial, &initial, &stream)?);
    drop(blocker.update(&occupied, &occupied, &stream)?);

    let extension = Array::from_f32(&[3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 2])?;
    let context = first.update(&extension, &extension, &stream)?;
    context.keys.async_eval(&stream)?;
    stream.synchronize()?;

    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn reads_an_offset_page_run_as_a_view_and_split_runs_natively() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let stream = stream(&pool)?;
    let mut split = cache(&pool, 0)?;
    let mut offset = cache(&pool, 0)?;
    let initial = values(&[1.0, 2.0])?;
    drop(split.update(&initial, &initial, &stream)?);
    drop(offset.update(&initial, &initial, &stream)?);
    let mode = PagedContextMode::NativeIfFragmented;

    let token = values(&[3.0, 4.0])?;
    let view = offset.update_for_attention_mode(&token, &token, &stream, 1, mode)?;
    let extension = Array::from_f32(&[3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 2])?;
    let native = split.update_for_attention_mode(&extension, &extension, &stream, 1, mode)?;

    assert_ne!(offset.first_physical_page(), Some(0));
    assert!(view.paged.is_none());
    assert_eq!(view.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(
        native
            .paged
            .as_ref()
            .is_some_and(crate::engine::kv::PagedKvContext::is_fragmented)
    );
    Ok(())
}

#[test]
fn separates_incompatible_layers_and_retains_arenas_in_pool() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let stream = stream(&pool)?;
    let value = values(&[1.0, 2.0])?;
    {
        let mut first = cache(&pool, 0)?;
        let mut other_layer = cache(&pool, 1)?;
        drop(first.update(&value, &value, &stream)?);
        drop(other_layer.update(&value, &value, &stream)?);
        assert!(!first.shares_paged_arena(&other_layer));
        assert_eq!(pool.resident_arenas()?, 2);
    }
    assert_eq!(pool.resident_arenas()?, 2);
    Ok(())
}

#[test]
fn surviving_model_keeps_the_arena_after_its_creator_unloads() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let first_stream = stream(&pool)?;
    let mut first = cache(&pool, 0)?;
    let first_value = values(&[1.0, 2.0])?;
    let first_context = first.update(&first_value, &first_value, &first_stream)?;
    first_context.keys.async_eval(&first_stream)?;
    first_stream.synchronize()?;

    let second_stream = stream(&pool)?;
    let mut second = cache(&pool, 0)?;
    let second_value = values(&[3.0, 4.0])?;
    drop(second.update(&second_value, &second_value, &second_stream)?);
    second_stream.synchronize()?;
    drop(first);
    drop(first_stream);

    let extension = values(&[5.0, 6.0])?;
    let context = second.update(&extension, &extension, &second_stream)?;
    context.keys.async_eval(&second_stream)?;
    second_stream.synchronize()?;

    assert_eq!(context.keys.to_vec_f32(&second_stream)?, vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(pool.resident_arenas()?, 1);
    Ok(())
}

#[test]
fn shares_prefix_pages_across_distinct_runtime_block_tables() -> Result<()> {
    use runtime::kv::BlockId;

    let pool = Arc::new(PagedArenaPool::default());
    let stream = stream(&pool)?;
    let mut owner = cache(&pool, 0)?;
    owner.bind_blocks(&[BlockId(0), BlockId(1)], None, &stream)?;
    let prefix = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
    let written = owner.update(&prefix, &prefix, &stream)?;
    written.keys.async_eval(&stream)?;
    stream.synchronize()?;

    let mut branch = cache(&pool, 0)?;
    branch.bind_blocks(&[BlockId(0), BlockId(2)], Some(2), &stream)?;

    let current = owner.update(&values(&[5.0, 6.0])?, &values(&[5.0, 6.0])?, &stream)?;
    let diverged = branch.update(&values(&[7.0, 8.0])?, &values(&[7.0, 8.0])?, &stream)?;
    let current_scratch = current
        .paged
        .as_ref()
        .ok_or_else(|| {
            crate::engine::Error::InvalidModel("current paged context is missing".into())
        })?
        .scratch();
    let branch_scratch = diverged
        .paged
        .as_ref()
        .ok_or_else(|| {
            crate::engine::Error::InvalidModel("branch paged context is missing".into())
        })?
        .scratch();
    assert!(!std::ptr::eq(current_scratch, branch_scratch));
    current.keys.async_eval(&stream)?;
    diverged.keys.async_eval(&stream)?;
    stream.synchronize()?;

    assert_eq!(current.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(diverged.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0]);
    Ok(())
}

fn stream(pool: &Arc<PagedArenaPool>) -> Result<Stream> {
    Stream::new_gpu_with_config_and_pool(Arc::default(), Arc::clone(pool))
}

fn cache(pool: &Arc<PagedArenaPool>, layer: usize) -> Result<KvCache> {
    KvCache::new_paged_with_pool(2, 2, KvPageFormat::Native, Arc::clone(pool), layer)
}

fn values(values: &[f32]) -> Result<Array> {
    Array::from_f32(values, &[1, 1, 1, 2])
}
