use std::sync::Arc;

use super::{
    super::{BatchAttentionExecution, candidates, compatible_groups, forward, profile},
    support,
};
use crate::engine::{Array, Error, KvContext, Result, Stream};

#[test]
fn profiles_uniform_multi_token_prefill_without_paged_candidates() -> Result<()> {
    let query = Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
    let context = KvContext {
        keys: Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?,
        values: Array::from_f32(&[1.0; 4], &[1, 1, 2, 2])?,
        paged: None,
        mask: None,
    };
    let key = profile::key(&[&query, &query], &[&context, &context], true)?
        .ok_or_else(|| Error::InvalidModel("uniform prefill profile is missing".into()))?;
    assert_eq!(key.sequence, 2);
    assert_eq!(
        candidates(key, true, true),
        [BatchAttentionExecution::Rows, BatchAttentionExecution::Batched]
    );
    Ok(())
}

#[test]
fn variable_paged_contexts_exclude_incompatible_view_candidates() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let query = Array::from_f32(&[0.0; 32], &[1, 1, 1, 32])?;
    let first = support::paged_context(1_025, 32, 13, &stream)?;
    let second = support::paged_context(1_041, 32, 17, &stream)?;
    let queries = [&query, &query];
    let contexts = [&first, &second];
    assert_eq!(compatible_groups(&queries, &contexts, false)?, [vec![0, 1]]);
    let key = profile::key(&queries, &contexts, false)?
        .ok_or_else(|| Error::InvalidModel("variable paged profile is missing".into()))?;
    assert!(!key.view);
    assert_eq!(
        candidates(key, true, true),
        [BatchAttentionExecution::PagedRows, BatchAttentionExecution::PagedBatched12,]
    );
    Ok(())
}

#[test]
fn runtime_profiles_an_unseen_multi_token_prefill_shape() -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.warmup_iterations = 1;
    config.tuning.measurement_iterations = 1;
    config.tuning.startup_budget_ms = 10_000;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let query = Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
    let context = KvContext {
        keys: Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?,
        values: Array::from_f32(&[1.0; 4], &[1, 1, 2, 2])?,
        paged: None,
        mask: None,
    };
    let queries = [&query, &query];
    let contexts = [&context, &context];
    let key = profile::key(&queries, &contexts, true)?
        .ok_or_else(|| Error::InvalidModel("uniform prefill profile is missing".into()))?;
    stream
        .tuner
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .finish_startup();

    let outputs = forward(&queries, &contexts, 1.0, true, &stream)?
        .ok_or_else(|| Error::InvalidModel("prefill tuning output is missing".into()))?;
    Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0, &stream)?.async_eval()?;
    stream.synchronize()?;

    assert!(
        stream
            .tuner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .batch_attention_decision(key)
            .is_some()
    );
    Ok(())
}
