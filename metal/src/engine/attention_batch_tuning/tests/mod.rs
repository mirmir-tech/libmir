use std::sync::Arc;

mod batched_paged;
mod prefill;
mod support;

use support::{assert_outputs_close, native_decode_context, paged_context, patterned};

use super::{
    BatchAttentionExecution, candidates, compatible_groups, execute, fallback, forward, paged,
    profile,
};
use crate::engine::{Array, Error, KvCache, KvContext, PagedContextMode, Result, Stream};

#[test]
fn batched_attention_matches_independent_rows() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let first_query = Array::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 1, 2])?;
    let second_query = Array::from_f32(&[0.0, 1.0, 0.0, 1.0], &[1, 2, 1, 2])?;
    let first = context(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[10.0, 1.0, 20.0, 2.0, 30.0, 3.0])?;
    let second = context(&[0.0, 1.0, 1.0, 0.0, 0.0, 2.0], &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0])?;
    let queries = [&first_query, &second_query];
    let contexts = [&first, &second];
    let rows = execute(BatchAttentionExecution::Rows, &queries, &contexts, 0.5, false, &stream)?;
    let batched =
        execute(BatchAttentionExecution::Batched, &queries, &contexts, 0.5, false, &stream)?;
    let rows = Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, &stream)?;
    let batched = Array::concatenate(&batched.iter().collect::<Vec<_>>(), 0, &stream)?;
    batched.async_eval()?;
    stream.synchronize()?;
    let expected = rows.to_vec_f32()?;
    let actual = batched.to_vec_f32()?;
    assert_eq!(expected.len(), actual.len());
    assert!(expected.iter().zip(actual).all(|(left, right)| (left - right).abs() < 1.0e-5));
    Ok(())
}

#[test]
fn paged_executions_match_gathered_rows() -> Result<()> {
    const CONTEXT: usize = 1_024;
    const HEAD_DIM: usize = 32;
    let stream = Stream::new_gpu()?;
    let query_values = (0..4 * HEAD_DIM).map(|index| patterned(index, 11)).collect::<Vec<_>>();
    let query = Array::from_f32(&query_values, &[1, 4, 1, i32::try_from(HEAD_DIM)?])?;
    let first = paged_context(CONTEXT, HEAD_DIM, 17, &stream)?;
    let second = paged_context(CONTEXT, HEAD_DIM, 29, &stream)?;
    let queries = [&query, &query];
    let contexts = [&first, &second];
    for execution in [BatchAttentionExecution::PagedRows, BatchAttentionExecution::PagedBatched] {
        let expected =
            execute(BatchAttentionExecution::Rows, &queries, &contexts, 0.125, false, &stream)?;
        let actual = execute(execution, &queries, &contexts, 0.125, false, &stream)?;
        assert_outputs_close(&expected, &actual, &stream)?;
    }
    Ok(())
}

#[test]
fn batched_paged_matches_rows_for_shared_arena() -> Result<()> {
    const CONTEXT: usize = 1_024;
    const HEAD_DIM: usize = 32;
    let stream = Stream::new_gpu()?;
    let values = (0..CONTEXT * HEAD_DIM).map(|index| patterned(index, 19)).collect::<Vec<_>>();
    let keys =
        Array::from_f32(&values, &[1, 1, i32::try_from(CONTEXT)?, i32::try_from(HEAD_DIM)?])?;
    let values = Array::from_f32(
        &values.iter().rev().copied().collect::<Vec<_>>(),
        &[1, 1, i32::try_from(CONTEXT)?, i32::try_from(HEAD_DIM)?],
    )?;
    let mut base = KvCache::new_paged(CONTEXT * 2, 16)?;
    base.update_for_attention_mode(&keys, &values, &stream, 0, PagedContextMode::Both)?;
    let mut first_cache = base.snapshot_at(CONTEXT)?;
    let mut second_cache = base.snapshot_at(CONTEXT)?;
    let update = Array::from_f32(&[0.25; HEAD_DIM], &[1, 1, 1, i32::try_from(HEAD_DIM)?])?;
    let first = first_cache.update_for_attention_mode(
        &update,
        &update,
        &stream,
        0,
        PagedContextMode::Both,
    )?;
    let second = second_cache.update_for_attention_mode(
        &update,
        &update,
        &stream,
        0,
        PagedContextMode::Both,
    )?;
    let query_values = (0..4 * HEAD_DIM).map(|index| patterned(index, 23)).collect::<Vec<_>>();
    let query = Array::from_f32(&query_values, &[1, 4, 1, i32::try_from(HEAD_DIM)?])?;
    let queries = [&query, &query];
    let contexts = [&first, &second];
    assert!(paged::batchable(&contexts));
    let expected =
        execute(BatchAttentionExecution::Rows, &queries, &contexts, 0.125, false, &stream)?;
    let actual = execute(
        BatchAttentionExecution::PagedBatched,
        &queries,
        &contexts,
        0.125,
        false,
        &stream,
    )?;
    assert_outputs_close(&expected, &actual, &stream)
}

#[test]
fn profiles_only_uniform_view_contexts() -> Result<()> {
    let query = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let first = context(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1.0; 6])?;
    let second = KvContext {
        keys: Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?,
        values: Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?,
        paged: None,
        mask: None,
    };
    assert!(profile::key(&[&query, &query], &[&first, &first], false)?.is_some());
    assert!(profile::key(&[&query, &query], &[&first, &second], false)?.is_none());
    Ok(())
}

#[test]
fn partitions_outlier_contexts_without_discarding_the_compatible_batch() -> Result<()> {
    let query = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let common = context(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1.0; 6])?;
    let outlier = KvContext {
        keys: Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?,
        values: Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?,
        paged: None,
        mask: None,
    };
    let groups =
        compatible_groups(&[&query, &query, &query], &[&common, &outlier, &common], false)?;
    assert_eq!(groups, vec![vec![0, 2], vec![1]]);
    Ok(())
}

#[test]
fn fragmented_pages_retain_measured_view_candidates() -> Result<()> {
    let query = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let context = context(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1.0; 6])?;
    let mut key = profile::key(&[&query, &query], &[&context, &context], false)?
        .ok_or_else(|| Error::InvalidModel("uniform batch key is missing".into()))?;
    assert_eq!(
        candidates(key, false, false),
        vec![BatchAttentionExecution::Rows, BatchAttentionExecution::Batched]
    );
    key.fragmented = true;
    assert_eq!(
        candidates(key, false, false),
        vec![BatchAttentionExecution::Rows, BatchAttentionExecution::Batched]
    );
    assert_eq!(
        candidates(key, true, false),
        vec![
            BatchAttentionExecution::Rows,
            BatchAttentionExecution::Batched,
            BatchAttentionExecution::PagedRows,
        ]
    );
    assert_eq!(
        candidates(key, true, true),
        vec![
            BatchAttentionExecution::Rows,
            BatchAttentionExecution::Batched,
            BatchAttentionExecution::PagedRows,
            BatchAttentionExecution::PagedBatched,
        ]
    );
    key.view = false;
    assert_eq!(
        candidates(key, true, true),
        vec![BatchAttentionExecution::PagedRows, BatchAttentionExecution::PagedBatched]
    );
    key.view = true;
    key.head_dim = 128;
    key.query_heads = 32;
    key.kv_heads = 8;
    assert_eq!(fallback(key, true), BatchAttentionExecution::PagedRows);
    key.head_dim = 512;
    key.query_heads = 16;
    key.kv_heads = 2;
    assert_eq!(fallback(key, true), BatchAttentionExecution::PagedRows);
    key.context_bucket = 8_192;
    assert!(profile::prefer_paged_batched(key, true));
    key.fragmented = false;
    assert_eq!(fallback(key, true), BatchAttentionExecution::Rows);
    Ok(())
}

#[test]
fn groups_native_paged_rows_by_page_span_without_requiring_a_view() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let query = Array::from_f32(&[0.0; 32], &[1, 1, 1, 32])?;
    let first = native_decode_context(1_025, 32, 13, &stream)?;
    let second = native_decode_context(1_041, 32, 17, &stream)?;
    let queries = [&query, &query];
    let contexts = [&first, &second];
    assert_eq!(compatible_groups(&queries, &contexts, false)?, [vec![0, 1]]);
    let key = profile::key(&queries, &contexts, false)?
        .ok_or_else(|| Error::InvalidModel("native paged profile is missing".into()))?;
    assert!(!key.view);
    assert!(paged::batchable(&contexts));
    assert_eq!(
        candidates(key, true, true),
        [BatchAttentionExecution::PagedRows, BatchAttentionExecution::PagedBatched]
    );
    Ok(())
}

#[test]
fn startup_profiles_once_and_reuses_the_shape_decision() -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.warmup_iterations = 1;
    config.tuning.measurement_iterations = 1;
    config.tuning.startup_budget_ms = 10_000;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let query = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let context = context(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1.0; 6])?;
    let queries = [&query, &query];
    let contexts = [&context, &context];
    let key = profile::key(&queries, &contexts, false)?
        .ok_or_else(|| Error::InvalidModel("uniform batch key is missing".into()))?;
    let first = forward(&queries, &contexts, 0.5, false, &stream)?
        .ok_or_else(|| Error::InvalidModel("profiled output is missing".into()))?;
    let selected = stream
        .tuner
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .batch_attention_decision(key);
    assert!(selected.is_some());
    let second = forward(&queries, &contexts, 0.5, false, &stream)?
        .ok_or_else(|| Error::InvalidModel("cached output is missing".into()))?;
    assert_eq!(first.len(), second.len());
    Ok(())
}

fn context(keys: &[f32], values: &[f32]) -> Result<KvContext> {
    Ok(KvContext {
        keys: Array::from_f32(keys, &[1, 1, 3, 2])?,
        values: Array::from_f32(values, &[1, 1, 3, 2])?,
        paged: None,
        mask: None,
    })
}
