mod measurement;
mod paged;
mod profile;
#[cfg(test)]
mod tests;
mod view;

use std::time::Instant;

pub(super) use profile::{BatchAttentionKey, compatible_groups};
use runtime::tuning::{TuningMode, select_fastest_candidate};

use self::profile::fallback;
use super::{Array, KvContext, Result, Stream};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum BatchAttentionExecution {
    Rows,
    Batched,
    PagedRows,
    PagedBatched,
}

pub(super) fn forward(
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Option<Vec<Array>>> {
    let Some(key) = profile::key(queries, contexts, causal)? else {
        return Ok(None);
    };
    let paged = contexts.iter().all(|context| context.paged.is_some());
    let paged_batched = paged::batchable(contexts);
    if profile::prefer_paged_batched(key, paged_batched) {
        return execute(
            BatchAttentionExecution::PagedBatched,
            queries,
            contexts,
            scale,
            causal,
            stream,
        )
        .map(Some);
    }
    let fallback = fallback(key, paged);
    let action = {
        let tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if tuner.config().mode == TuningMode::Disabled {
            Some(fallback)
        } else if let Some(execution) = tuner.batch_attention_decision(key) {
            Some(execution)
        } else if tuner.config().mode == TuningMode::Startup
            && (tuner.batch_attention_budget_available(key.causal)
                || paged && tuner.batch_attention_runtime_budget_available(key.causal))
        {
            None
        } else {
            Some(fallback)
        }
    };
    action.map_or_else(
        || {
            let started = Instant::now();
            tune(key, queries, contexts, scale, causal, paged_batched, stream)
                .or_else(|error| {
                    stream
                        .tuner
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .record_batch_attention(
                            key,
                            BatchAttentionExecution::Rows,
                            started.elapsed(),
                        );
                    tracing::warn!(
                        target: "libmir::metal::tuning",
                        %error,
                        "Metal packed-attention tuning failed; retaining row execution"
                    );
                    execute(BatchAttentionExecution::Rows, queries, contexts, scale, causal, stream)
                })
                .map(Some)
        },
        |execution| execute(execution, queries, contexts, scale, causal, stream).map(Some),
    )
}

fn tune(
    key: BatchAttentionKey,
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    causal: bool,
    paged_batched: bool,
    stream: &Stream,
) -> Result<Vec<Array>> {
    let started = Instant::now();
    let candidates =
        candidates(key, contexts.iter().all(|context| context.paged.is_some()), paged_batched);
    let timings = candidates
        .iter()
        .copied()
        .map(|execution| measurement::measure(execution, queries, contexts, scale, causal, stream))
        .collect::<Result<Vec<_>>>()?;
    let fastest = timings
        .iter()
        .enumerate()
        .min_by_key(|(_, time)| *time)
        .map_or(0, |(index, _)| index);
    let selected = select_fastest_candidate(
        fastest,
        0,
        &timings,
        stream.config().tuning.minimum_improvement_bps,
    );
    let execution = candidates[selected];
    {
        let mut tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        tuner.record_batch_attention(key, execution, started.elapsed());
        tuner.persist();
    }
    tracing::info!(
        target: "libmir::metal::tuning",
        ?execution,
        ?key,
        timings_us = ?timings
            .iter()
            .map(|duration| duration.as_secs_f64() * 1_000_000.0)
            .collect::<Vec<_>>(),
        "selected Metal packed-attention execution profile"
    );
    execute(execution, queries, contexts, scale, causal, stream)
}

fn candidates(
    key: BatchAttentionKey,
    paged: bool,
    paged_batched: bool,
) -> Vec<BatchAttentionExecution> {
    let mut candidates = Vec::new();
    if key.view {
        candidates.extend([BatchAttentionExecution::Rows, BatchAttentionExecution::Batched]);
    }
    if paged && key.sequence == 1 {
        candidates.push(BatchAttentionExecution::PagedRows);
    }
    if paged_batched && key.sequence == 1 {
        candidates.push(BatchAttentionExecution::PagedBatched);
    }
    candidates
}

fn execute(
    execution: BatchAttentionExecution,
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Vec<Array>> {
    execute_measured(execution, queries, contexts, scale, causal, stream, false)
}

fn execute_measured(
    execution: BatchAttentionExecution,
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
    refresh_fragmented: bool,
) -> Result<Vec<Array>> {
    match execution {
        BatchAttentionExecution::Rows => queries
            .iter()
            .zip(contexts)
            .map(|(query, context)| {
                let (keys, values) = view::attention(context, stream, refresh_fragmented)?;
                query.scaled_dot_product_attention(&keys, &values, scale, causal, stream)
            })
            .collect(),
        BatchAttentionExecution::Batched => {
            batched(queries, contexts, scale, causal, stream, refresh_fragmented)
        },
        BatchAttentionExecution::PagedRows => paged::rows(queries, contexts, scale, stream),
        BatchAttentionExecution::PagedBatched => paged::batched(queries, contexts, scale, stream),
    }
}

fn batched(
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
    refresh_fragmented: bool,
) -> Result<Vec<Array>> {
    let queries = Array::concatenate(queries, 0, stream)?;
    let views = contexts
        .iter()
        .map(|context| view::attention(context, stream, refresh_fragmented))
        .collect::<Result<Vec<_>>>()?;
    let keys = views.iter().map(|(keys, _)| keys).collect::<Vec<_>>();
    let values = views.iter().map(|(_, values)| values).collect::<Vec<_>>();
    let keys = Array::concatenate(&keys, 0, stream)?;
    let values = Array::concatenate(&values, 0, stream)?;
    let output = queries.scaled_dot_product_attention(&keys, &values, scale, causal, stream)?;
    let shape = output.shape()?;
    (0..shape[0])
        .map(|row| {
            output.slice(
                &[usize::try_from(row)?, 0, 0, 0],
                &[
                    usize::try_from(row + 1)?,
                    usize::try_from(shape[1])?,
                    usize::try_from(shape[2])?,
                    usize::try_from(shape[3])?,
                ],
                stream,
            )
        })
        .collect()
}
