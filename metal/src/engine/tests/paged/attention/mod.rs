mod benchmark;

use std::env;

use crate::engine::{
    Array, Error, KvCache, ModelTensors, PagedAttention, Result, Stream,
    attention::PagedAttentionScratch,
};

#[test]
fn paged_sdpa_matches_contiguous_gqa() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let queries = Array::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 1, 2])?;
    let keys = Array::from_f32(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1, 1, 3, 2])?;
    let values = Array::from_f32(&[10.0, 1.0, 20.0, 2.0, 30.0, 3.0], &[1, 1, 3, 2])?;
    let key_pages = Array::from_f32(&[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[1, 2, 2, 2])?;
    let value_pages = Array::from_f32(&[30.0, 3.0, 0.0, 0.0, 10.0, 1.0, 20.0, 2.0], &[1, 2, 2, 2])?;
    let page_table = Array::from_u32(&[1, 0], &[2])?;
    let page_dependency = Array::from_u32(&[3], &[1])?;

    let expected = queries.scaled_dot_product_attention(&keys, &values, 0.5, false, &stream)?;
    let paged = PagedAttention {
        key_pages: &key_pages,
        value_pages: &value_pages,
        key_scales: None,
        value_scales: None,
        page_table: &page_table,
        page_dependency: &page_dependency,
        page_size: 2,
        context_tokens: 3,
    };
    let actual = queries.paged_scaled_dot_product_attention(paged, 0.5, &stream)?;

    actual.async_eval(&stream)?;
    stream.synchronize()?;
    let expected = expected.to_vec_f32(&stream)?;
    let actual = actual.to_vec_f32(&stream)?;
    assert_close(&expected, &actual);
    Ok(())
}

#[test]
fn paged_sdpa_matches_gemma_global_attention_shape() -> Result<()> {
    const CONTEXT: usize = 128;
    const HEAD_DIM: usize = 512;
    const KV_HEADS: usize = 2;
    const QUERY_HEADS: usize = 16;
    const PAGE_SIZE: usize = 16;
    let stream = Stream::new_gpu()?;
    let query = patterned(QUERY_HEADS * HEAD_DIM, 17);
    let key_values = patterned(KV_HEADS * CONTEXT * HEAD_DIM, 31);
    let value_data = patterned(KV_HEADS * CONTEXT * HEAD_DIM, 47);
    let queries = Array::from_f32(&query, &[1, 16, 1, 512])?;
    let keys = Array::from_f32(&key_values, &[1, 2, 128, 512])?;
    let values = Array::from_f32(&value_data, &[1, 2, 128, 512])?;
    let key_pages = Array::from_f32(&key_values, &[2, 8, 16, 512])?;
    let value_pages = Array::from_f32(&value_data, &[2, 8, 16, 512])?;
    let page_table = Array::from_u32(&(0..8).collect::<Vec<_>>(), &[8])?;
    let dependency = Array::from_u32(&[128], &[1])?;
    let expected = queries.scaled_dot_product_attention(&keys, &values, 1.0, false, &stream)?;
    let actual = queries.paged_scaled_dot_product_attention(
        PagedAttention {
            key_pages: &key_pages,
            value_pages: &value_pages,
            key_scales: None,
            value_scales: None,
            page_table: &page_table,
            page_dependency: &dependency,
            page_size: PAGE_SIZE,
            context_tokens: CONTEXT,
        },
        1.0,
        &stream,
    )?;
    actual.async_eval(&stream)?;
    stream.synchronize()?;
    assert_approx(&expected.to_vec_f32(&stream)?, &actual.to_vec_f32(&stream)?, 0.002);
    Ok(())
}

#[test]
fn paged_two_pass_matches_contiguous_attention() -> Result<()> {
    const CONTEXT: usize = 1_024;
    const HEAD_DIM: usize = 64;
    const KV_HEADS: usize = 2;
    const QUERY_HEADS: usize = 4;
    const PAGES: usize = CONTEXT / 16;
    let stream = Stream::new_gpu()?;
    let query = patterned(QUERY_HEADS * HEAD_DIM, 17);
    let keys = patterned(KV_HEADS * CONTEXT * HEAD_DIM, 31);
    let values = patterned(KV_HEADS * CONTEXT * HEAD_DIM, 47);
    let query = Array::from_f32(&query, &[1, 4, 1, 64])?;
    let contiguous_keys = Array::from_f32(&keys, &[1, 2, 1_024, 64])?;
    let contiguous_values = Array::from_f32(&values, &[1, 2, 1_024, 64])?;
    let key_pages = Array::from_f32(&keys, &[2, 64, 16, 64])?;
    let value_pages = Array::from_f32(&values, &[2, 64, 16, 64])?;
    let page_table = Array::from_u32(&(0..u32::try_from(PAGES)?).collect::<Vec<_>>(), &[64])?;
    let dependency = Array::from_u32(&[u32::try_from(CONTEXT)?], &[1])?;
    let expected = query.scaled_dot_product_attention(
        &contiguous_keys,
        &contiguous_values,
        1.0,
        false,
        &stream,
    )?;
    let paged = PagedAttention {
        key_pages: &key_pages,
        value_pages: &value_pages,
        key_scales: None,
        value_scales: None,
        page_table: &page_table,
        page_dependency: &dependency,
        page_size: 16,
        context_tokens: CONTEXT,
    };
    let scratch = PagedAttentionScratch::default();
    let first =
        query.paged_scaled_dot_product_attention_with_scratch(paged, &scratch, 1.0, &stream)?;
    let second = query.paged_scaled_dot_product_attention_with_scratch(
        PagedAttention {
            key_pages: &key_pages,
            value_pages: &value_pages,
            key_scales: None,
            value_scales: None,
            page_table: &page_table,
            page_dependency: &dependency,
            page_size: 16,
            context_tokens: CONTEXT,
        },
        &scratch,
        1.0,
        &stream,
    )?;
    second.async_eval(&stream)?;
    stream.synchronize()?;
    let expected = expected.to_vec_f32(&stream)?;
    assert_approx(&expected, &first.to_vec_f32(&stream)?, 1.0e-4);
    assert_approx(&expected, &second.to_vec_f32(&stream)?, 1.0e-4);
    Ok(())
}

#[test]
fn writes_persistent_kv_pages_on_the_mlx_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_paged(2, 2)?;
    let first_keys = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let first_values = Array::from_f32(&[10.0, 1.0], &[1, 1, 1, 2])?;
    drop(cache.update(&first_keys, &first_values, &stream)?);
    let second_keys = Array::from_f32(&[0.0, 1.0, 2.0, 0.0], &[1, 1, 2, 2])?;
    let second_values = Array::from_f32(&[20.0, 2.0, 30.0, 3.0], &[1, 1, 2, 2])?;
    let context = cache.update(&second_keys, &second_values, &stream)?;
    let queries = Array::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 1, 2])?;
    let keys = Array::from_f32(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1, 1, 3, 2])?;
    let values = Array::from_f32(&[10.0, 1.0, 20.0, 2.0, 30.0, 3.0], &[1, 1, 3, 2])?;

    let expected = queries.scaled_dot_product_attention(&keys, &values, 1.0, false, &stream)?;
    let actual = queries
        .scaled_dot_product_attention(&context.keys, &context.values, 1.0, false, &stream)?;
    actual.async_eval(&stream)?;
    stream.synchronize()?;
    assert_eq!(cache.offset()?, 3);
    assert_close(&expected.to_vec_f32(&stream)?, &actual.to_vec_f32(&stream)?);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn paged_bfloat16_gemma_global_attention_matches_contiguous() -> Result<()> {
    let root = env::var_os("MIRMIR_BENCH_MODEL")
        .or_else(|| env::var_os("MODEL"))
        .ok_or_else(|| Error::InvalidModel("set MIRMIR_BENCH_MODEL or MODEL".into()))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let reference = tensors.get("language_model.model.norm.weight")?;
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_paged(128, 16)?;
    let query = Array::from_f32(&patterned(16 * 512, 17), &[1, 16, 1, 512])?
        .astype_like(&reference, &stream)?;
    let keys = Array::from_f32(&patterned(2 * 128 * 512, 31), &[1, 2, 128, 512])?
        .astype_like(&reference, &stream)?;
    let values = Array::from_f32(&patterned(2 * 128 * 512, 47), &[1, 2, 128, 512])?
        .astype_like(&reference, &stream)?;
    let context = cache.update(&keys, &values, &stream)?;
    let Some(paged) = context.paged else {
        return Err(Error::InvalidModel("paged K/V storage was not created".into()));
    };
    paged.page_dependency.async_eval(&stream)?;
    let expected = query.scaled_dot_product_attention(&keys, &values, 1.0, false, &stream)?;
    let flattened =
        query.scaled_dot_product_attention(&context.keys, &context.values, 1.0, false, &stream)?;
    let actual = query.paged_scaled_dot_product_attention(paged.attention(), 1.0, &stream)?;
    actual.async_eval(&stream)?;
    stream.synchronize()?;
    assert_approx(&expected.to_vec_f32(&stream)?, &flattened.to_vec_f32(&stream)?, 0.002);
    assert_approx(&expected.to_vec_f32(&stream)?, &actual.to_vec_f32(&stream)?, 0.002);
    Ok(())
}

fn assert_close(expected: &[f32], actual: &[f32]) {
    assert_approx(expected, actual, 1e-4);
}

fn assert_approx(expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(expected.len(), actual.len());
    let maximum = expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum < tolerance, "maximum absolute error {maximum} exceeds {tolerance}");
}

fn patterned(length: usize, multiplier: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let value = u8::try_from(index.wrapping_mul(multiplier) % 127).unwrap_or_default();
            (f32::from(value) - 63.0) / 127.0
        })
        .collect()
}
