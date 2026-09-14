use crate::engine::{
    Array, KvContext, Result, Stream,
    attention::{AttentionBias, AttentionRequest},
};

pub(super) fn row(
    query: &Array,
    context: &KvContext,
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    AttentionRequest::new(query, context, scale, causal, AttentionBias::None)?.execute(stream)
}

pub(super) fn batch(
    queries: &Array,
    contexts: &[KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    #[cfg(test)]
    if !causal
        && queries.shape()?[2] == 1
        && contexts.iter().all(|context| context.paged.is_none())
        && stream.config().diagnostics.history_batching == crate::config::HistoryBatching::Rows
    {
        crate::engine::probe::history::record_row_batch();
        return separate_rows(queries, contexts, scale, causal, stream);
    }
    if contexts.iter().any(|context| context.paged.is_some()) {
        #[cfg(test)]
        let shape = queries.shape()?;
        #[cfg(test)]
        if let Some(execution) = stream.config().diagnostics.attention_candidate {
            let rows = (0..contexts.len())
                .map(|row| {
                    queries.slice(
                        &[row, 0, 0, 0],
                        &[
                            row + 1,
                            usize::try_from(shape[1])?,
                            usize::try_from(shape[2])?,
                            usize::try_from(shape[3])?,
                        ],
                        stream,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let outputs = crate::engine::attention_batch_tuning::execute(
                execution,
                &rows.iter().collect::<Vec<_>>(),
                &contexts.iter().collect::<Vec<_>>(),
                scale,
                causal,
                stream,
            )?;
            return Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0, stream);
        }
        return separate_rows(queries, contexts, scale, causal, stream);
    }
    let keys = contexts.iter().map(|context| &context.keys).collect::<Vec<_>>();
    let values = contexts.iter().map(|context| &context.values).collect::<Vec<_>>();
    let joined_keys = Array::concatenate(&keys, 0, stream)?;
    let joined_values = Array::concatenate(&values, 0, stream)?;
    let output = queries
        .scaled_dot_product_attention(&joined_keys, &joined_values, scale, causal, stream)?;
    #[cfg(test)]
    crate::engine::probe::history::joined(
        queries,
        &keys,
        &values,
        [&joined_keys, &joined_values],
        &output,
        scale,
        causal,
    )?;
    Ok(output)
}

fn separate_rows(
    queries: &Array,
    contexts: &[KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    let shape = queries.shape()?;
    let rows = contexts
        .iter()
        .enumerate()
        .map(|(index, context)| {
            let query = queries.slice(
                &[index, 0, 0, 0],
                &[
                    index + 1,
                    usize::try_from(shape[1])?,
                    usize::try_from(shape[2])?,
                    usize::try_from(shape[3])?,
                ],
                stream,
            )?;
            row(&query, context, scale, causal, stream)
        })
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)
}
