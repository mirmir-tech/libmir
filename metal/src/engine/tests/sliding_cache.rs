use crate::engine::{Array, KvCache, Result, Stream};

#[test]
fn updates_sliding_cache_with_a_chunk_across_the_window() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_with_window(3, Some(3))?;
    let first = Array::from_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1])?;
    drop(cache.update(&first, &first, &stream)?);
    let update = Array::from_f32(&[4.0, 5.0], &[1, 1, 2, 1])?;
    let context = cache.update(&update, &update, &stream)?;
    context.keys.async_eval(&stream)?;
    stream.synchronize()?;

    assert_eq!(cache.offset()?, 5);
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![2.0, 3.0, 4.0, 5.0]);
    let mask = context.mask.as_ref().ok_or(crate::engine::Error::NullHandle("sliding mask"))?;
    assert_eq!(mask.to_vec_u32(&stream)?, vec![1, 1, 1, 0, 0, 1, 1, 1]);

    let query = Array::from_f32(&[0.0, 0.0], &[1, 1, 2, 1])?;
    let keys = Array::from_f32(&[0.0; 4], &[1, 1, 4, 1])?;
    let output =
        query.masked_scaled_dot_product_attention(&keys, &context.values, 1.0, mask, &stream)?;
    assert_eq!(output.to_vec_f32(&stream)?, vec![3.0, 4.0]);

    let token = Array::from_f32(&[6.0], &[1, 1, 1, 1])?;
    let context = cache.update(&token, &token, &stream)?;
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn preserves_temporal_order_across_consecutive_sliding_chunks() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_with_window(3, Some(3))?;
    let first = Array::from_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1])?;
    drop(cache.update(&first, &first, &stream)?);
    let second = Array::from_f32(&[4.0, 5.0], &[1, 1, 2, 1])?;
    drop(cache.update(&second, &second, &stream)?);
    let third = Array::from_f32(&[6.0, 7.0], &[1, 1, 2, 1])?;
    let context = cache.update(&third, &third, &stream)?;

    assert_eq!(cache.offset()?, 7);
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![4.0, 5.0, 6.0, 7.0]);
    let mask = context.mask.as_ref().ok_or(crate::engine::Error::NullHandle("sliding mask"))?;
    assert_eq!(mask.to_vec_u32(&stream)?, vec![1, 1, 1, 0, 0, 1, 1, 1]);
    Ok(())
}
