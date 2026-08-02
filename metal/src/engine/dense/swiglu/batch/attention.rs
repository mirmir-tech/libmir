use super::super::{attention, config::DenseSwiGluLayerConfig, weights::AttentionWeights};
use crate::engine::{
    Array, Error, FusedAttention, KvCache, PagedContextMode, Result, Stream,
    attention_batch_tuning, native_paged_attention_mode, paged_attention_min_context,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn packed_attention(
    input: &Array,
    weights: &AttentionWeights,
    fused: Option<&FusedAttention>,
    config: DenseSwiGluLayerConfig,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    let batch = i32::try_from(caches.len())?;
    let sequence = *input
        .shape()?
        .get(1)
        .ok_or_else(|| Error::InvalidModel("packed input has no sequence axis".into()))?;
    let fused = (caches.len() <= 2).then_some(fused).flatten();
    let projections = fused.map_or_else(
        || {
            Ok::<_, Error>((
                weights.query.forward(input, stream)?,
                weights.key.forward(input, stream)?,
                weights.value.forward(input, stream)?,
            ))
        },
        |fused| {
            let output = fused.forward(input, stream)?;
            Ok::<_, Error>((
                output.query,
                output.key,
                output
                    .value
                    .ok_or_else(|| Error::InvalidModel("fused attention omitted values".into()))?,
            ))
        },
    )?;
    let (queries, keys, values) = projections;
    let queries = queries.reshape(&[batch, sequence, config.heads, config.head_dim], stream)?;
    let queries =
        attention::normalize(queries, weights.query_norm.as_ref(), config.rms_norm_eps, stream)?;
    let keys = keys.reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?;
    let keys = attention::normalize(keys, weights.key_norm.as_ref(), config.rms_norm_eps, stream)?;
    let values = values
        .reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    let rows = packed_attention_rows(
        &queries,
        &keys,
        &values,
        caches,
        AttentionRows {
            weights,
            config,
            positions,
            causal,
            stream,
        },
    )?;
    let rows = rows.iter().collect::<Vec<_>>();
    let output = Array::concatenate(&rows, 0, stream)?.transpose(&[0, 2, 1, 3], stream)?;
    let width = config.heads * config.head_dim;
    weights
        .output
        .forward(&output.reshape(&[batch, sequence, width], stream)?, stream)
}

fn packed_attention_rows(
    queries: &Array,
    keys: &Array,
    values: &Array,
    caches: &mut [&mut KvCache],
    context: AttentionRows<'_>,
) -> Result<Vec<Array>> {
    let AttentionRows {
        weights,
        config,
        positions,
        causal,
        stream,
    } = context;
    let tune_paged = caches.len() > 1;
    let prepared = caches
        .iter_mut()
        .enumerate()
        .map(|(row, cache)| {
            let position = positions[row];
            let query = sequence_row_slice(queries, row, config.heads, config.head_dim, stream)?;
            let query = attention::rope_layout(
                &query,
                weights.rope_frequencies.as_ref(),
                config,
                position,
                stream,
            )?;
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
                    config.heads,
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
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = grouped_attention(&prepared, config.attention_scale, causal, stream)?;
    prepared
        .into_iter()
        .enumerate()
        .map(|(row, (query, context, mode))| {
            if let Some(output) = outputs[row].take() {
                return Ok(output);
            }
            let view = mode == PagedContextMode::Both
                && context
                    .paged
                    .as_ref()
                    .is_none_or(|paged| paged.key_scales.is_none() && paged.value_scales.is_none());
            if view || context.paged.is_none() {
                return query.scaled_dot_product_attention(
                    &context.keys,
                    &context.values,
                    config.attention_scale,
                    causal,
                    stream,
                );
            }
            let paged = context
                .paged
                .ok_or_else(|| Error::InvalidModel("native paged context is missing".into()))?;
            query.paged_scaled_dot_product_attention_with_scratch(
                paged.attention(),
                paged.scratch(),
                config.attention_scale,
                stream,
            )
        })
        .collect()
}

fn grouped_attention(
    prepared: &[(Array, crate::engine::KvContext, PagedContextMode)],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Vec<Option<Array>>> {
    let queries = prepared.iter().map(|(query, _, _)| query).collect::<Vec<_>>();
    let contexts = prepared.iter().map(|(_, context, _)| context).collect::<Vec<_>>();
    let groups = attention_batch_tuning::compatible_groups(&queries, &contexts, causal)?;
    let mut outputs = std::iter::repeat_with(|| None)
        .take(prepared.len())
        .collect::<Vec<Option<Array>>>();
    for rows in groups.iter().filter(|rows| rows.len() > 1) {
        let group_queries = rows.iter().map(|&row| queries[row]).collect::<Vec<_>>();
        let group_contexts = rows.iter().map(|&row| contexts[row]).collect::<Vec<_>>();
        let Some(group_outputs) = attention_batch_tuning::forward(
            &group_queries, &group_contexts, scale, causal, stream,
        )?
        else {
            continue;
        };
        for (&row, output) in rows.iter().zip(group_outputs) {
            outputs[row] = Some(output);
        }
    }
    Ok(outputs)
}

#[derive(Clone, Copy)]
struct AttentionRows<'a> {
    weights: &'a AttentionWeights,
    config: DenseSwiGluLayerConfig,
    positions: &'a [i32],
    causal: bool,
    stream: &'a Stream,
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
