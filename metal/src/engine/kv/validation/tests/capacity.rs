use std::sync::Arc;

use runtime::kv::BlockId;

use crate::engine::{Array, Error, KvCache, KvPageFormat, PagedArenaPool, Result, Stream};

#[test]
fn bind_blocks_rejects_out_of_range_block_ids_and_reads_shared_prefix_directly() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pool = Arc::new(PagedArenaPool::default());
    let first = Array::from_f32(&[1.0, 0.5, -0.5, 0.25, 0.25, 0.5, 0.75, 1.0], &[1, 1, 2, 4])?;
    let next = Array::from_f32(&[0.0, 1.0, 0.0, 0.5], &[1, 1, 1, 4])?;

    let mut producer =
        KvCache::new_paged_with_pool_capacity(4, 2, KvPageFormat::Native, 4, Arc::clone(&pool), 0)?;
    let out_of_range = producer.bind_blocks(&[BlockId(4)], None, &stream);
    assert!(matches!(
        out_of_range,
        Err(Error::KvBlockOutOfRange { layer: 0, block: 4, capacity: 4 })
    ));

    // Producer writes 2 tokens into runtime BlockId(2).
    producer.bind_blocks(&[BlockId(2)], None, &stream)?;
    drop(producer.update(&first, &first, &stream)?);
    pool.eval_with_graph_roots(&[], &stream)?;
    stream.synchronize()?;
    drop(producer);

    // Arena stays resident in pool after producer drops so consumer reuses
    // BlockId(2).
    assert_eq!(pool.resident_arenas()?, 1);

    let mut consumer =
        KvCache::new_paged_with_pool_capacity(4, 2, KvPageFormat::Native, 4, Arc::clone(&pool), 0)?;
    consumer.bind_blocks(&[BlockId(2), BlockId(0)], Some(2), &stream)?;
    assert_eq!(consumer.offset()?, 2);
    let ctx = consumer.update(&next, &next, &stream)?;
    assert_eq!(consumer.offset()?, 3);
    assert_eq!(
        ctx.keys.to_vec_f32(&stream)?,
        [1.0, 0.5, -0.5, 0.25, 0.25, 0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 0.5]
    );
    Ok(())
}

#[test]
fn rejects_write_exceeding_bound_block_table() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pool = Arc::new(PagedArenaPool::default());
    let two_tokens = Array::from_f32(&[1.0; 8], &[1, 1, 2, 4])?;
    let one_token = Array::from_f32(&[2.0; 4], &[1, 1, 1, 4])?;
    let mut cache =
        KvCache::new_paged_with_pool_capacity(4, 2, KvPageFormat::Native, 4, Arc::clone(&pool), 0)?;
    cache.bind_blocks(&[BlockId(1)], None, &stream)?;
    drop(cache.update(&two_tokens, &two_tokens, &stream)?);
    assert!(!cache.can_write_token(2));
    let overflow = cache.update(&one_token, &one_token, &stream);
    assert!(matches!(
        overflow,
        Err(Error::KvPageCapacity { layer: 0, required: 2, maximum: 1 })
    ));
    assert_eq!(cache.offset()?, 2);
    Ok(())
}
