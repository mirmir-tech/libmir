use std::sync::Arc;

use crate::engine::{Array, Error, KvCache, KvPageFormat, PagedArenaPool, Result, Stream};

fn cache(format: KvPageFormat, pool: &Arc<PagedArenaPool>) -> Result<KvCache> {
    KvCache::new_paged_with_pool_capacity(4, 4, format, 1, Arc::clone(pool), 0)
}

#[test]
fn rejected_first_page_allocation_remains_inactive_and_can_retry() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let token = Array::from_f32(&[1.0, 0.5, -0.5, 0.25], &[1, 1, 1, 4])?;
    for format in [KvPageFormat::Native, KvPageFormat::Int8PerTokenHead] {
        let pool = Arc::new(PagedArenaPool::default());
        let mut owner = cache(format, &pool)?;
        drop(owner.update(&token, &token, &stream)?);
        pool.eval_with_graph_roots(&[], &stream)?;
        let mut waiting = cache(format, &pool)?;
        assert!(waiting.update(&token, &token, &stream).is_err());
        assert_eq!(waiting.offset()?, 0);
        assert_eq!(waiting.physical_page_count(), 0);
        assert!(!waiting.pages.as_ref().ok_or(Error::NullHandle("pages"))?.active());
        owner.reset()?;
        drop(waiting.update(&token, &token, &stream)?);
        pool.eval_with_graph_roots(&[], &stream)?;
        assert_eq!(waiting.offset()?, 1);
        waiting.reset()?;
        assert_eq!(pool.resident_arenas()?, 0);
    }
    Ok(())
}

#[test]
fn rejected_promotion_preserves_contiguous_tokens_and_can_retry() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let first = Array::from_f32(&[1.0, 0.5, -0.5, 0.25], &[1, 1, 1, 4])?;
    let next = Array::from_f32(&[0.0, 1.0, 0.0, 0.5], &[1, 1, 1, 4])?;
    for format in [KvPageFormat::Native, KvPageFormat::Int8PerTokenHead] {
        let pool = Arc::new(PagedArenaPool::default());
        let mut owner = cache(format, &pool)?;
        drop(owner.update(&first, &first, &stream)?);
        let mut waiting = cache(format, &pool)?;
        drop(waiting.update_with_page_min_context(&first, &first, &stream, 2)?);
        pool.eval_with_graph_roots(&[], &stream)?;
        let before = waiting.keys.as_ref().ok_or(Error::NullHandle("keys"))?.to_vec_f32(&stream)?;
        let capacity = waiting.capacity;
        assert!(waiting.update_with_page_min_context(&next, &next, &stream, 2).is_err());
        assert_eq!(waiting.offset()?, 1, "failed promotion consumed a token");
        assert_eq!(waiting.capacity, capacity);
        assert!(!waiting.pages.as_ref().ok_or(Error::NullHandle("pages"))?.active());
        assert_eq!(
            waiting.keys.as_ref().ok_or(Error::NullHandle("keys"))?.to_vec_f32(&stream)?,
            before
        );
        owner.reset()?;
        let context = waiting.update_with_page_min_context(&next, &next, &stream, 2)?;
        assert_eq!(waiting.offset()?, 2);
        assert!(waiting.keys.is_none() && waiting.values.is_none());
        if format == KvPageFormat::Native {
            assert_eq!(
                context.keys.to_vec_f32(&stream)?,
                [1.0, 0.5, -0.5, 0.25, 0.0, 1.0, 0.0, 0.5]
            );
        }
        pool.eval_with_graph_roots(&[], &stream)?;
        drop(context);
        waiting.reset()?;
        assert_eq!(pool.resident_arenas()?, 0);
    }
    Ok(())
}

#[test]
fn rejected_cow_preserves_shared_page_and_retries_after_release() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let token = Array::from_f32(&[1.0, 0.5, -0.5, 0.25], &[1, 1, 1, 4])?;
    for format in [KvPageFormat::Native, KvPageFormat::Int8PerTokenHead] {
        let pool = Arc::new(PagedArenaPool::default());
        let mut owner = cache(format, &pool)?;
        drop(owner.update(&token, &token, &stream)?);
        pool.eval_with_graph_roots(&[], &stream)?;
        let snapshot = owner.snapshot_at(1)?;
        let page = owner.first_physical_page();
        assert!(owner.update(&token, &token, &stream).is_err());
        assert_eq!(owner.offset()?, 1);
        assert_eq!(owner.first_physical_page(), page);
        assert_eq!(snapshot.first_physical_page(), page);
        drop(snapshot);
        drop(owner.update(&token, &token, &stream)?);
        pool.eval_with_graph_roots(&[], &stream)?;
        assert_eq!(owner.offset()?, 2);
        assert_eq!(owner.first_physical_page(), page);
        owner.reset()?;
        assert_eq!(pool.resident_arenas()?, 0);
    }
    Ok(())
}
