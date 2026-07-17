use super::super::{HybridMoeLayerConfig, attention, weights::AttentionWeights};
use crate::engine::{
    Array, FusedAttention, FusedKeyValue, KvCache, Result, Stream, native_paged_attention_mode,
    paged_attention_min_context,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn packed_attention(
    input: &Array,
    weights: &AttentionWeights,
    config: HybridMoeLayerConfig,
    fused_attention: Option<&FusedAttention>,
    fused_key_value: Option<&FusedKeyValue>,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let batch = i32::try_from(caches.len())?;
    let (queries, raw_keys, raw_values) =
        projections(input, weights, fused_attention, fused_key_value, stream)?;
    let queries = queries.reshape(&[batch, 1, config.attention_heads, config.head_dim], stream)?;
    let queries = weights.query_norm.apply(&queries, config.rms_norm_eps, stream)?;
    let raw_keys = raw_keys.reshape(&[batch, 1, config.kv_heads, config.head_dim], stream)?;
    let values = values(&raw_keys, raw_values, batch, config, stream)?;
    let keys = weights.key_norm.apply(&raw_keys, config.rms_norm_eps, stream)?;
    let rows = attention_rows(
        &queries,
        &keys,
        &values,
        caches,
        AttentionRows { weights, config, positions, stream },
    )?;
    let rows = rows.iter().collect::<Vec<_>>();
    let output = Array::concatenate(&rows, 0, stream)?.transpose(&[0, 2, 1, 3], stream)?;
    let width = config.attention_heads * config.head_dim;
    weights.output.forward(&output.reshape(&[batch, 1, width], stream)?, stream)
}

fn projections(
    input: &Array,
    weights: &AttentionWeights,
    fused_attention: Option<&FusedAttention>,
    fused_key_value: Option<&FusedKeyValue>,
    stream: &Stream,
) -> Result<(Array, Array, Option<Array>)> {
    if let Some(fused) = fused_attention {
        let output = fused.forward(input, stream)?;
        return Ok((output.query, output.key, output.value));
    }
    let (keys, values) = match fused_key_value {
        Some(fused) => {
            let (key, value) = fused.forward(input, stream)?;
            (key, Some(value))
        },
        None => (
            weights.key.forward(input, stream)?,
            weights.value.as_ref().map(|value| value.forward(input, stream)).transpose()?,
        ),
    };
    Ok((weights.query.forward(input, stream)?, keys, values))
}

fn values(
    raw_keys: &Array,
    raw_values: Option<Array>,
    batch: i32,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<Array> {
    let values = if config.use_k_eq_v {
        raw_keys.rms_norm_unit(config.rms_norm_eps, stream)?
    } else {
        raw_values
            .ok_or_else(|| {
                crate::engine::Error::InvalidModel("missing hybrid MoE value projection".into())
            })?
            .reshape(&[batch, 1, config.kv_heads, config.head_dim], stream)?
            .rms_norm_unit(config.rms_norm_eps, stream)?
    };
    values.transpose(&[0, 2, 1, 3], stream)
}

fn attention_rows(
    queries: &Array,
    keys: &Array,
    values: &Array,
    caches: &mut [&mut KvCache],
    context: AttentionRows<'_>,
) -> Result<Vec<Array>> {
    caches
        .iter_mut()
        .enumerate()
        .map(|(row, cache)| attention_row(queries, keys, values, cache, row, context))
        .collect()
}

fn attention_row(
    queries: &Array,
    keys: &Array,
    values: &Array,
    cache: &mut KvCache,
    row: usize,
    context: AttentionRows<'_>,
) -> Result<Array> {
    let AttentionRows { weights, config, positions, stream } = context;
    let position = positions[row];
    let query = sequence_row_slice(queries, row, config.attention_heads, config.head_dim, stream)?;
    let query = attention::rope_layout(
        &query,
        weights.rope_frequencies.as_ref(),
        config,
        position,
        stream,
    )?;
    let key = sequence_row_slice(keys, row, config.kv_heads, config.head_dim, stream)?;
    let key =
        attention::rope_layout(&key, weights.rope_frequencies.as_ref(), config, position, stream)?;
    let value = row_slice(values, row, config.kv_heads, config.head_dim, stream)?;
    let mode = native_paged_attention_mode(
        config.head_dim,
        config.attention_heads,
        config.kv_heads,
        usize::try_from(position)? + 1,
        stream.config().cache.force_native_paged_attention,
    );
    let context = cache.update_for_attention_mode(
        &key,
        &value,
        stream,
        paged_attention_min_context(stream),
        mode,
    )?;
    if let Some(paged) = context.paged {
        query.paged_scaled_dot_product_attention_with_scratch(
            paged.attention(),
            paged.scratch(),
            1.0,
            stream,
        )
    } else if let Some(mask) = context.mask.as_ref() {
        query.masked_scaled_dot_product_attention(&context.keys, &context.values, 1.0, mask, stream)
    } else {
        query.scaled_dot_product_attention(&context.keys, &context.values, 1.0, false, stream)
    }
}

#[derive(Clone, Copy)]
struct AttentionRows<'a> {
    weights: &'a AttentionWeights,
    config: HybridMoeLayerConfig,
    positions: &'a [i32],
    stream: &'a Stream,
}

fn row_slice(
    input: &Array,
    row: usize,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(heads)?, 1, usize::try_from(head_dim)?],
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
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, 1, usize::try_from(heads)?, usize::try_from(head_dim)?],
        stream,
    )
}
