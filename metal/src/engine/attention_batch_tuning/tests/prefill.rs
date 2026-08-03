use super::{
    super::{BatchAttentionExecution, candidates, compatible_groups, profile},
    support,
};
use crate::engine::{Array, KvContext, Result, Stream};

#[test]
fn profiles_uniform_multi_token_prefill_without_paged_candidates() -> Result<()> {
    let query = Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
    let context = KvContext {
        keys: Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?,
        values: Array::from_f32(&[1.0; 4], &[1, 1, 2, 2])?,
        paged: None,
        mask: None,
    };
    let key = profile::key(&[&query, &query], &[&context, &context], true)?.unwrap();
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
    let key = profile::key(&queries, &contexts, false)?.unwrap();
    assert!(!key.view);
    assert_eq!(
        candidates(key, true, true),
        [BatchAttentionExecution::PagedRows, BatchAttentionExecution::PagedBatched]
    );
    Ok(())
}
