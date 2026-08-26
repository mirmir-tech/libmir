use crate::engine::{Array, Error, KvCache, KvPageFormat, PagedContextMode, Result, Stream};

#[test]
fn int8_paged_prefill_matches_native_causal_attention() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let queries = Array::from_f32(
        &[
            0.5, -0.25, 0.75, 0.125, 0.25, 0.5, -0.5, 0.75, -0.75, 0.25, 0.5, -0.125, -0.25, 0.75,
            0.5, -0.5, 0.5, -0.75, 0.25, 0.125, 0.75, 0.5, -0.25, 0.25,
        ],
        &[1, 2, 3, 4],
    )?;
    let keys = Array::from_f32(
        &[1.0, -0.5, 0.25, 0.75, 0.5, 1.0, -0.25, 0.125, -0.75, 0.5, 1.0, -0.5],
        &[1, 1, 3, 4],
    )?;
    let values = Array::from_f32(
        &[2.0, 1.0, -1.0, 0.5, 1.0, -2.0, 0.25, 0.75, -0.5, 1.5, 2.0, -1.0],
        &[1, 1, 3, 4],
    )?;
    let expected = queries.scaled_dot_product_attention(&keys, &values, 0.5, true, &stream)?;
    let mut cache = KvCache::new_paged_with_format(16, 2, KvPageFormat::Int8PerTokenHead)?;
    let context =
        cache.update_for_attention_mode(&keys, &values, &stream, 0, PagedContextMode::Native)?;
    let paged = context.paged.ok_or(Error::NullHandle("INT8 paged context"))?;
    assert_eq!(paged.key_pages.native().dtype()?, mirtal::DType::Uint32);
    assert!(paged.key_scales.is_some());
    assert!(paged.value_scales.is_some());
    let actual = queries.paged_scaled_dot_product_attention(paged.attention(), 0.5, &stream)?;
    actual.async_eval(&stream)?;
    stream.synchronize()?;
    assert_close(&expected.to_vec_f32(&stream)?, &actual.to_vec_f32(&stream)?, 0.025);
    Ok(())
}

#[test]
fn int8_pages_preserve_scales_across_copy_on_write() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let initial_keys =
        Array::from_f32(&[1.0, 0.5, -0.25, 0.75, -0.5, 1.0, 0.25, -0.75], &[1, 1, 2, 4])?;
    let initial_values =
        Array::from_f32(&[2.0, -1.0, 0.5, 1.0, -0.5, 1.5, 2.0, -1.0], &[1, 1, 2, 4])?;
    let mut left = int8_cache()?;
    left.update_for_attention_mode(
        &initial_keys,
        &initial_values,
        &stream,
        0,
        PagedContextMode::Native,
    )?;
    let mut right = left.snapshot_at(2)?;
    let query = Array::from_f32(&[0.5, -0.25, 0.75, 0.125], &[1, 1, 1, 4])?;
    let left_key = Array::from_f32(&[0.25, 0.5, 1.0, -0.5], &[1, 1, 1, 4])?;
    let left_value = Array::from_f32(&[1.0, 2.0, -0.5, 0.25], &[1, 1, 1, 4])?;
    let right_key = Array::from_f32(&[-1.0, 0.25, 0.5, 0.75], &[1, 1, 1, 4])?;
    let right_value = Array::from_f32(&[-2.0, 0.5, 1.0, 1.5], &[1, 1, 1, 4])?;
    let left_actual = append_and_attend(&mut left, &query, &left_key, &left_value, &stream)?;
    let right_actual = append_and_attend(&mut right, &query, &right_key, &right_value, &stream)?;
    let left_keys = Array::concatenate(&[&initial_keys, &left_key], 2, &stream)?;
    let left_values = Array::concatenate(&[&initial_values, &left_value], 2, &stream)?;
    let right_keys = Array::concatenate(&[&initial_keys, &right_key], 2, &stream)?;
    let right_values = Array::concatenate(&[&initial_values, &right_value], 2, &stream)?;
    let left_expected =
        query.scaled_dot_product_attention(&left_keys, &left_values, 1.0, false, &stream)?;
    let right_expected =
        query.scaled_dot_product_attention(&right_keys, &right_values, 1.0, false, &stream)?;
    left_actual.async_eval(&stream)?;
    right_actual.async_eval(&stream)?;
    stream.synchronize()?;
    assert_close(&left_expected.to_vec_f32(&stream)?, &left_actual.to_vec_f32(&stream)?, 0.035);
    assert_close(&right_expected.to_vec_f32(&stream)?, &right_actual.to_vec_f32(&stream)?, 0.035);
    Ok(())
}

fn int8_cache() -> Result<KvCache> {
    KvCache::new_paged_with_format(16, 2, KvPageFormat::Int8PerTokenHead)
}

fn append_and_attend(
    cache: &mut KvCache,
    query: &Array,
    key: &Array,
    value: &Array,
    stream: &Stream,
) -> Result<Array> {
    let context =
        cache.update_for_attention_mode(key, value, stream, 0, PagedContextMode::Native)?;
    let paged = context.paged.ok_or(Error::NullHandle("INT8 paged context"))?;
    query.paged_scaled_dot_product_attention(paged.attention(), 1.0, stream)
}

fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(expected.len(), actual.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            (expected - actual).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}
