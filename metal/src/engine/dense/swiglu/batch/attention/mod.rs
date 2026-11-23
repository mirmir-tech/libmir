use super::super::{attention, config::DenseSwiGluLayerConfig, weights::AttentionWeights};
use crate::engine::{
    Array, Error, FusedAttention, KvCache, PagedContextMode, Result, Stream,
    paged_attention_min_context,
};

mod rows;

use rows::{grouped_attention, row_slice, sequence_row_slice};

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
    // Capped scores and position-dependent attention run row by row.
    if config.attention_softcap.is_some() || config.positional() {
        return super::super::positional::packed(
            input, weights, config, caches, positions, causal, stream,
        );
    }
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
    let queries = attention::heads(
        queries,
        weights.query_norm.as_ref(),
        weights,
        config,
        [batch, sequence, config.heads],
        stream,
    )?;
    let keys = attention::heads(
        keys,
        weights.key_norm.as_ref(),
        weights,
        config,
        [batch, sequence, config.kv_heads],
        stream,
    )?;
    let values = values
        .reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    if let Some(output) = super::history::persistent_attention(
        &queries, &keys, &values, caches, weights, config, positions, causal, stream,
    )? {
        let output = output.transpose(&[0, 2, 1, 3], stream)?;
        let width = config.heads * config.head_dim;
        return weights
            .output
            .forward(&output.reshape(&[batch, sequence, width], stream)?, stream);
    }
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
            let mode =
                super::row_context_mode(cache, config, position, sequence, tune_paged, stream)?;
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
    // Rows whose sliding cache masks a bulk write run alone with that mask.
    let mut outputs = if prepared.iter().any(|(_, context, _)| context.mask.is_some()) {
        std::iter::repeat_with(|| None).take(prepared.len()).collect()
    } else {
        grouped_attention(&prepared, config.attention_scale, causal, stream)?
    };
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
            if let Some(mask) = context.mask.as_ref() {
                return query.masked_scaled_dot_product_attention(
                    &context.keys,
                    &context.values,
                    config.attention_scale,
                    mask,
                    stream,
                );
            }
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

#[derive(Clone, Copy)]
struct AttentionRows<'a> {
    weights: &'a AttentionWeights,
    config: DenseSwiGluLayerConfig,
    positions: &'a [i32],
    causal: bool,
    stream: &'a Stream,
}
