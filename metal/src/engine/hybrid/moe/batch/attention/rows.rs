//! Each packed row's queries over its own cache context, grouped where rows
//! can share one attention call.

use super::{super::super::attention, AttentionRows};
use crate::engine::{
    Array, KvCache, KvContext, PagedContextMode, Result, Stream, attention::tuning,
    native_paged_attention_mode, paged_attention_min_context,
};

pub(super) fn concatenate(rows: &[Array], stream: &Stream) -> Result<Array> {
    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)
}

/// Row `row`'s rotated `[1, heads, S, D]` queries.
pub(super) fn row_query(queries: &Array, row: usize, context: AttentionRows<'_>) -> Result<Array> {
    let AttentionRows { weights, config, positions, stream, .. } = context;
    let query = sequence_row_slice(queries, row, config.attention_heads, config.head_dim, stream)?;
    attention::rope_layout(
        &query,
        weights.rope_frequencies.as_ref(),
        config,
        positions[row],
        stream,
    )
}

/// Each row's rotated queries, its cache context after appending its keys
/// and values, and the mode the context was read in.
pub(super) fn prepare_rows(
    queries: &Array,
    keys: &Array,
    values: &Array,
    caches: &mut [&mut KvCache],
    context: AttentionRows<'_>,
) -> Result<Vec<(Array, KvContext, PagedContextMode)>> {
    let AttentionRows { weights, config, positions, stream, .. } = context;
    let tune_paged = caches.len() > 1;
    caches
        .iter_mut()
        .enumerate()
        .map(|(row, cache)| {
            let position = positions[row];
            let query = row_query(queries, row, context)?;
            let key = sequence_row_slice(keys, row, config.kv_heads, config.head_dim, stream)?;
            let key = attention::rope_layout(
                &key,
                weights.rope_frequencies.as_ref(),
                config,
                position,
                stream,
            )?;
            let value = row_slice(values, row, config.kv_heads, config.head_dim, stream)?;
            let sequence = usize::try_from(query.shape()?[2])?;
            let mode = if sequence == 1 && tune_paged {
                PagedContextMode::Both
            } else if sequence == 1 {
                native_paged_attention_mode(
                    config.head_dim,
                    config.attention_heads,
                    config.kv_heads,
                    usize::try_from(position)? + 1,
                    stream.config().cache.force_native_paged_attention,
                )
            } else {
                PagedContextMode::View
            };
            let context = cache.update_for_attention_mode(
                &key,
                &value,
                stream,
                paged_attention_min_context(stream),
                mode,
            )?;
            Ok((query, context, mode))
        })
        .collect()
}

/// Each row's queries over its context: grouped where rows can share one
/// call, else one by one.
pub(super) fn attend_rows(
    rows: &[(Array, &KvContext, PagedContextMode)],
    causal: bool,
    stream: &Stream,
) -> Result<Vec<Array>> {
    let mut outputs = grouped_attention(rows, causal, stream)?;
    rows.iter()
        .enumerate()
        .map(|(row, (query, context, mode))| {
            if let Some(output) = outputs[row].take() {
                return Ok(output);
            }
            let view = *mode == PagedContextMode::Both
                && context.paged.as_ref().is_none_or(|paged| !paged.is_quantized());
            match &context.paged {
                Some(paged) if !view => query.paged_scaled_dot_product_attention_with_scratch(
                    paged.attention(),
                    paged.scratch(),
                    1.0,
                    stream,
                ),
                _ => query.scaled_dot_product_attention(
                    &context.keys, &context.values, 1.0, causal, stream,
                ),
            }
        })
        .collect()
}

fn grouped_attention(
    prepared: &[(Array, &KvContext, PagedContextMode)],
    causal: bool,
    stream: &Stream,
) -> Result<Vec<Option<Array>>> {
    let queries = prepared.iter().map(|(query, _, _)| query).collect::<Vec<_>>();
    let contexts = prepared.iter().map(|(_, context, _)| *context).collect::<Vec<_>>();
    let groups = tuning::batch::compatible_groups(&queries, &contexts, causal)?;
    let mut outputs = std::iter::repeat_with(|| None).take(prepared.len()).collect::<Vec<_>>();
    for rows in groups.iter().filter(|rows| rows.len() > 1) {
        let group_queries = rows.iter().map(|&row| queries[row]).collect::<Vec<_>>();
        let group_contexts = rows.iter().map(|&row| contexts[row]).collect::<Vec<_>>();
        let Some(group_outputs) =
            tuning::batch::forward(&group_queries, &group_contexts, 1.0, causal, stream)?
        else {
            continue;
        };
        for (&row, output) in rows.iter().zip(group_outputs) {
            outputs[row] = Some(output);
        }
    }
    Ok(outputs)
}

fn row_slice(
    input: &Array,
    row: usize,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    let sequence = usize::try_from(input.shape()?[2])?;
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(heads)?, sequence, usize::try_from(head_dim)?],
        stream,
    )
}

fn sequence_row_slice(
    input: &Array,
    row: usize,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    let sequence = usize::try_from(input.shape()?[1])?;
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, sequence, usize::try_from(heads)?, usize::try_from(head_dim)?],
        stream,
    )
}
