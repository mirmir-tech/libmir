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

    // The rewind margin keeps older keys; the mask hides them.
    assert_eq!(cache.offset()?, 5);
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = context.mask.as_ref().ok_or(crate::engine::Error::NullHandle("sliding mask"))?;
    assert_eq!(mask.to_vec_u32(&stream)?, vec![0, 1, 1, 1, 0, 0, 0, 1, 1, 1]);

    let query = Array::from_f32(&[0.0, 0.0], &[1, 1, 2, 1])?;
    let keys = Array::from_f32(&[0.0; 5], &[1, 1, 5, 1])?;
    let output =
        query.masked_scaled_dot_product_attention(&keys, &context.values, 1.0, mask, &stream)?;
    assert_eq!(output.to_vec_f32(&stream)?, vec![3.0, 4.0]);

    let token = Array::from_f32(&[6.0], &[1, 1, 1, 1])?;
    let context = cache.update(&token, &token, &stream)?;
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Decode compacts the prompt tail's margin into the window-sized ring.
    let token = Array::from_f32(&[7.0], &[1, 1, 1, 1])?;
    let context = cache.update(&token, &token, &stream)?;
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![7.0, 5.0, 6.0]);
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
    assert_eq!(context.keys.to_vec_f32(&stream)?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let mask = context.mask.as_ref().ok_or(crate::engine::Error::NullHandle("sliding mask"))?;
    assert_eq!(mask.to_vec_u32(&stream)?, vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]);
    Ok(())
}

#[test]
fn rewinds_a_sliding_snapshot_within_its_margin() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let tokens = |values: &[f32]| Array::from_f32(values, &[1, 1, i32::try_from(values.len())?, 1]);
    let mut cache = KvCache::new_with_window(3, Some(3))?;
    let first = (1..=80)
        .map(|value| f32::from(u8::try_from(value).unwrap_or(0)))
        .collect::<Vec<_>>();
    drop(cache.update(&tokens(&first)?, &tokens(&first)?, &stream)?);
    let chunk = tokens(&[81.0, 82.0, 83.0])?;
    drop(cache.update(&chunk, &chunk, &stream)?);
    // The chunk kept window - 1 + 64 earlier keys: positions 14..83 (values
    // 15..83), so a snapshot rewinds as far as position 16.
    assert_eq!(cache.prefix_floor(), 16);
    assert!(cache.snapshot_at(15).is_err());
    let mut rewound = cache.snapshot_at(20)?;
    let next = tokens(&[100.0, 101.0])?;
    let context = rewound.update(&next, &next, &stream)?;
    assert_eq!(rewound.offset()?, 22);
    assert_eq!(
        context.keys.to_vec_f32(&stream)?,
        [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 100.0, 101.0]
    );
    let mut single = cache.snapshot_at(21)?;
    let token = tokens(&[200.0])?;
    let context = single.update(&token, &token, &stream)?;
    assert_eq!(
        context.keys.to_vec_f32(&stream)?,
        [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 200.0]
    );
    Ok(())
}
