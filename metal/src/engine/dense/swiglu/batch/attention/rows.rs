//! Per-row slices of packed attention inputs, and the shared-shape groups
//! that attend in one call.

use crate::engine::{Array, PagedContextMode, Result, Stream, attention::tuning};

pub(super) fn grouped_attention(
    prepared: &[(Array, crate::engine::KvContext, PagedContextMode)],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Vec<Option<Array>>> {
    let queries = prepared.iter().map(|(query, _, _)| query).collect::<Vec<_>>();
    let contexts = prepared.iter().map(|(_, context, _)| context).collect::<Vec<_>>();
    let groups = tuning::batch::compatible_groups(&queries, &contexts, causal)?;
    let mut outputs = std::iter::repeat_with(|| None)
        .take(prepared.len())
        .collect::<Vec<Option<Array>>>();
    for rows in groups.iter().filter(|rows| rows.len() > 1) {
        let group_queries = rows.iter().map(|&row| queries[row]).collect::<Vec<_>>();
        let group_contexts = rows.iter().map(|&row| contexts[row]).collect::<Vec<_>>();
        let Some(group_outputs) =
            tuning::batch::forward(&group_queries, &group_contexts, scale, causal, stream)?
        else {
            continue;
        };
        for (&row, output) in rows.iter().zip(group_outputs) {
            outputs[row] = Some(output);
        }
    }
    Ok(outputs)
}

pub(super) fn row_slice(
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

pub(super) fn sequence_row_slice(
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
