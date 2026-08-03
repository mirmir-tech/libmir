use std::sync::Arc;

use crate::engine::{Array, KvCache, KvPageFormat, PagedArenaPool, Result, Stream};

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
    first_context.keys.async_eval()?;
    first_stream.synchronize()?;
    let second_context = second.update(&second_value, &second_value, &second_stream)?;
    second_context.keys.async_eval()?;
    second_stream.synchronize()?;

    assert!(first.shares_paged_arena(&second));
    assert_ne!(first.first_physical_page(), second.first_physical_page());
    assert_eq!(first_context.keys.to_vec_f32_on_stream(&first_stream)?, vec![1.0, 2.0]);
    assert_eq!(second_context.keys.to_vec_f32_on_stream(&second_stream)?, vec![3.0, 4.0]);
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
    context.keys.async_eval()?;
    stream.synchronize()?;

    assert_eq!(context.keys.to_vec_f32_on_stream(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn separates_incompatible_layers_and_releases_empty_arenas() -> Result<()> {
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
    assert_eq!(pool.resident_arenas()?, 0);
    Ok(())
}

#[test]
fn surviving_model_keeps_the_arena_after_its_creator_unloads() -> Result<()> {
    let pool = Arc::new(PagedArenaPool::default());
    let first_stream = stream(&pool)?;
    let mut first = cache(&pool, 0)?;
    let first_value = values(&[1.0, 2.0])?;
    let first_context = first.update(&first_value, &first_value, &first_stream)?;
    first_context.keys.async_eval()?;
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
    context.keys.async_eval()?;
    second_stream.synchronize()?;

    assert_eq!(context.keys.to_vec_f32_on_stream(&second_stream)?, vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(pool.resident_arenas()?, 1);
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
