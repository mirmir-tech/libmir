use std::{
    hint::black_box,
    io::Write,
    time::{Duration, Instant},
};

use crate::engine::{Array, Error, KvCache, KvContext, PagedContextMode, Result, Stream};

mod snapshot;

#[test]
fn writes_strided_prefill_kv_pages() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let source = Array::from_f32(&[1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 20.0, 0.0], &[1, 2, 2, 2])?;
    let keys = source.transpose(&[0, 2, 1, 3], &stream)?;
    let value_source =
        Array::from_f32(&[100.0, 0.0, 300.0, 0.0, 0.0, 100.0, 0.0, 200.0], &[1, 2, 2, 2])?;
    let values = value_source.transpose(&[0, 2, 1, 3], &stream)?;
    let mut cache = KvCache::new_paged(2, 2)?;
    let context = cache.update(&keys, &values, &stream)?;
    let Some(paged) = context.paged else {
        return Err(Error::InvalidModel("paged K/V storage was not created".into()));
    };
    paged.page_dependency.async_eval()?;
    stream.synchronize()?;
    let stored = paged.key_pages.to_vec_f32_on_stream(&stream)?;
    assert_eq!(&stored[..8], &[1.0, 0.0, 0.0, 1.0, 10.0, 0.0, 20.0, 0.0]);
    let stored = paged.value_pages.to_vec_f32_on_stream(&stream)?;
    assert_eq!(&stored[..8], &[100.0, 0.0, 0.0, 100.0, 300.0, 0.0, 0.0, 200.0]);

    let queries = Array::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 1, 2])?;
    let expected = queries.scaled_dot_product_attention(&keys, &values, 1.0, false, &stream)?;
    let actual = queries
        .scaled_dot_product_attention(&context.keys, &context.values, 1.0, false, &stream)?;
    let paged_actual =
        queries.paged_scaled_dot_product_attention(paged.attention(), 1.0, &stream)?;
    actual.async_eval()?;
    paged_actual.async_eval()?;
    stream.synchronize()?;
    let expected = expected.to_vec_f32()?;
    assert_eq!(expected, actual.to_vec_f32()?);
    assert!(
        expected
            .iter()
            .zip(paged_actual.to_vec_f32()?)
            .all(|(left, right)| { (left - right).abs() < 1.0e-4 })
    );
    Ok(())
}

#[test]
fn backfills_pages_when_the_context_reaches_its_threshold() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_paged(2, 2)?;
    let first = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let first_context = cache.update_with_page_min_context(&first, &first, &stream, 3)?;
    assert!(first_context.paged.is_none());

    let second = Array::from_f32(&[0.0, 1.0, 2.0, 0.0], &[1, 1, 2, 2])?;
    let context = cache.update_with_page_min_context(&second, &second, &stream, 3)?;
    let _paged = context
        .paged
        .ok_or_else(|| Error::InvalidModel("paged K/V backfill was not created".into()))?;
    let queries = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let expected_kv = Array::from_f32(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1, 1, 3, 2])?;
    let expected =
        queries.scaled_dot_product_attention(&expected_kv, &expected_kv, 1.0, false, &stream)?;
    let actual = queries
        .scaled_dot_product_attention(&context.keys, &context.values, 1.0, false, &stream)?;
    actual.async_eval()?;
    stream.synchronize()?;
    assert_eq!(expected.to_vec_f32()?, actual.to_vec_f32()?);
    Ok(())
}

#[test]
fn snapshots_paged_kv_pages_without_mutation() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_paged(2, 2)?;
    let first = Array::from_f32(&[1.0, 2.0], &[1, 1, 1, 2])?;
    drop(cache.update(&first, &first, &stream)?);
    let mut snapshot = cache.snapshot_at(1)?;

    let second = Array::from_f32(&[3.0, 4.0], &[1, 1, 1, 2])?;
    let current = cache.update(&second, &second, &stream)?;
    let alternate = Array::from_f32(&[5.0, 6.0], &[1, 1, 1, 2])?;
    let branch = snapshot.update(&alternate, &alternate, &stream)?;
    let current_scratch = current
        .paged
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("current paged context is missing".into()))?
        .scratch();
    let branch_scratch = branch
        .paged
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("snapshot paged context is missing".into()))?
        .scratch();
    assert!(!std::ptr::eq(current_scratch, branch_scratch));
    current.keys.async_eval()?;
    branch.keys.async_eval()?;
    stream.synchronize()?;

    assert_eq!(current.keys.to_vec_f32_on_stream(&stream)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(branch.keys.to_vec_f32_on_stream(&stream)?, vec![1.0, 2.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn selects_explicit_native_or_gathered_paged_attention() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut current = KvCache::new_paged(2, 2)?;
    let first = Array::from_f32(&[1.0, 2.0], &[1, 1, 1, 2])?;
    drop(current.update(&first, &first, &stream)?);
    let mut identity = current.snapshot_at(1)?;
    let next = Array::from_f32(&[3.0, 4.0], &[1, 1, 1, 2])?;

    let fragmented =
        current.update_for_attention_mode(&next, &next, &stream, 1, PagedContextMode::Native)?;
    let contiguous =
        identity.update_for_attention_mode(&next, &next, &stream, 1, PagedContextMode::View)?;

    assert!(fragmented.paged.is_some());
    assert!(contiguous.paged.is_none());
    assert_eq!(contiguous.keys.to_vec_f32_on_stream(&stream)?, vec![1.0, 2.0, 3.0, 4.0]);
    let query = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let expected_kv = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
    let expected =
        query.scaled_dot_product_attention(&expected_kv, &expected_kv, 1.0, false, &stream)?;
    let paged = fragmented
        .paged
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("fragmented paged context is missing".into()))?;
    assert!(paged.fragmented);
    let actual = query.paged_scaled_dot_product_attention(paged.attention(), 1.0, &stream)?;
    actual.async_eval()?;
    stream.synchronize()?;
    assert!(
        expected
            .to_vec_f32()?
            .iter()
            .zip(actual.to_vec_f32()?)
            .all(|(left, right)| (left - right).abs() < 1.0e-5)
    );
    Ok(())
}

#[test]
#[ignore = "synthetic GPU benchmark"]
fn benchmarks_page_backed_mlx_sdpa_decode() -> Result<()> {
    const CONTEXT: usize = 2_048;
    const HEAD_DIM: usize = 512;
    const KV_HEADS: usize = 2;
    const QUERY_HEADS: usize = 16;
    let stream = Stream::new_gpu()?;
    let context = kv_array(CONTEXT, KV_HEADS, HEAD_DIM, 0.001)?;
    let token = kv_array(1, KV_HEADS, HEAD_DIM, 0.001)?;
    let query = Array::from_f32(
        &vec![0.01; QUERY_HEADS * HEAD_DIM],
        &[1, i32::try_from(QUERY_HEADS)?, 1, i32::try_from(HEAD_DIM)?],
    )?;
    let mut contiguous = KvCache::new(256)?;
    let mut paged = KvCache::new_paged(256, 16)?;
    drop(contiguous.update_for_attention(&context, &context, &stream, 0)?);
    drop(paged.update_for_attention(&context, &context, &stream, 1)?);
    let contiguous_context = contiguous.update_for_attention(&token, &token, &stream, 0)?;
    let paged_context = paged.update_for_attention(&token, &token, &stream, 1)?;
    let contiguous_read_ms = measure_attention(&query, &contiguous_context, &stream)?;
    let page_backed_read_ms = measure_attention(&query, &paged_context, &stream)?;

    let contiguous_ms = measure_decode(&mut contiguous, &query, &token, &stream, 0)?;
    let paged_ms = measure_decode(&mut paged, &query, &token, &stream, 1)?;
    writeln!(
        std::io::stderr().lock(),
        "paged_store.benchmark: context={CONTEXT}, read_contiguous={contiguous_read_ms:.3}ms, read_page_backed={page_backed_read_ms:.3}ms, contiguous={contiguous_ms:.3}ms, page_backed={paged_ms:.3}ms"
    )?;
    Ok(())
}

fn kv_array(tokens: usize, heads: usize, dimensions: usize, value: f32) -> Result<Array> {
    Array::from_f32(
        &vec![value; tokens * heads * dimensions],
        &[1, i32::try_from(heads)?, i32::try_from(tokens)?, i32::try_from(dimensions)?],
    )
}

fn measure_decode(
    cache: &mut KvCache,
    query: &Array,
    token: &Array,
    stream: &Stream,
    page_min_context: usize,
) -> Result<f64> {
    const WARMUP: usize = 3;
    const ITERATIONS: usize = 20;
    for _ in 0..WARMUP {
        decode_step(cache, query, token, stream, page_min_context)?;
    }
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        decode_step(cache, query, token, stream, page_min_context)?;
    }
    Ok(milliseconds(started.elapsed()) / f64::from(u32::try_from(ITERATIONS)?))
}

fn measure_attention(query: &Array, context: &KvContext, stream: &Stream) -> Result<f64> {
    const WARMUP: usize = 3;
    const ITERATIONS: usize = 20;
    for _ in 0..WARMUP {
        attention_step(query, context, stream)?;
    }
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        attention_step(query, context, stream)?;
    }
    Ok(milliseconds(started.elapsed()) / f64::from(u32::try_from(ITERATIONS)?))
}

fn decode_step(
    cache: &mut KvCache,
    query: &Array,
    token: &Array,
    stream: &Stream,
    page_min_context: usize,
) -> Result<()> {
    let context = cache.update_for_attention(token, token, stream, page_min_context)?;
    attention_step(query, &context, stream)
}

fn attention_step(query: &Array, context: &KvContext, stream: &Stream) -> Result<()> {
    let output =
        query.scaled_dot_product_attention(&context.keys, &context.values, 1.0, false, stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    black_box(output);
    Ok(())
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
