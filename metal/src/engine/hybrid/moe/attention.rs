use super::{
    HybridMoeLayerConfig,
    weights::{AttentionWeights, KeyValueWeights, ProjectedKeyValues},
};
use crate::engine::{
    Array, FusedAttention, FusedKeyValue, KvCache, Result, RopeOptions, Stream,
    native_paged_attention_mode, paged_attention_min_context,
};

pub(super) struct DecodeContext<'a> {
    pub(super) cache: Option<&'a mut KvCache>,
    pub(super) position: i32,
    pub(super) causal: bool,
    pub(super) mask: Option<&'a Array>,
    pub(super) stream: &'a Stream,
}

pub(super) fn forward_decode(
    input: &Array,
    weights: &AttentionWeights,
    config: HybridMoeLayerConfig,
    fused_attention: Option<&FusedAttention>,
    fused_key_value: Option<&FusedKeyValue>,
    context: DecodeContext<'_>,
) -> Result<Array> {
    let DecodeContext { cache, position, causal, mask, stream } = context;
    let sequence = input.shape()?.get(1).copied().ok_or_else(|| {
        crate::engine::Error::InvalidModel("attention input has no sequence axis".into())
    })?;
    let fused = if sequence == 1 {
        (fused_attention, fused_key_value)
    } else {
        (None, None)
    };
    let Projected { queries, keys, values } =
        project(input, weights, config, fused, [1, sequence], stream)?;
    let queries =
        rope_layout(&queries, weights.rope_frequencies.as_ref(), config, position, stream)?;
    let keys = rope_layout(&keys, weights.rope_frequencies.as_ref(), config, position, stream)?;
    let output = match cache {
        Some(cache) => {
            let mode = if sequence == 1 {
                native_paged_attention_mode(
                    config.head_dim,
                    config.attention_heads,
                    config.kv_heads,
                    usize::try_from(position)? + 1,
                    stream.config().cache.force_native_paged_attention,
                )
            } else {
                crate::engine::PagedContextMode::View
            };
            let context = cache.update_for_attention_mode(
                &keys,
                &values,
                stream,
                paged_attention_min_context(stream),
                mode,
            )?;
            if let Some(paged) = context.paged {
                if sequence == 1 {
                    queries.paged_scaled_dot_product_attention_with_scratch(
                        paged.attention(),
                        paged.scratch(),
                        1.0,
                        stream,
                    )?
                } else {
                    unreachable!("multi-token attention always requests a cache view")
                }
            } else if let Some(mask) = mask.or(context.mask.as_ref()) {
                queries.masked_scaled_dot_product_attention(
                    &context.keys, &context.values, 1.0, mask, stream,
                )?
            } else {
                queries.scaled_dot_product_attention(
                    &context.keys, &context.values, 1.0, causal, stream,
                )?
            }
        },
        None => match mask {
            Some(mask) => {
                queries.masked_scaled_dot_product_attention(&keys, &values, 1.0, mask, stream)?
            },
            None => queries.scaled_dot_product_attention(&keys, &values, 1.0, causal, stream)?,
        },
    };
    let output = output.transpose(&[0, 2, 1, 3], stream)?;
    let output_width = config.attention_heads * config.head_dim;
    let output = output.reshape(&[1, sequence, output_width], stream)?;
    weights.output.forward(&output, stream)
}

/// Normalized, unrotated `[B, S, heads, D]` queries and `[B, S, kv_heads,
/// D]` keys, and `[B, kv_heads, S, D]` values.
pub(super) struct Projected {
    pub(super) queries: Array,
    pub(super) keys: Array,
    pub(super) values: Array,
}

/// The layer's queries, and its keys and values: projected (and published
/// to the layers sharing them), or the earlier layer's of the same step.
pub(super) fn project(
    input: &Array,
    weights: &AttentionWeights,
    config: HybridMoeLayerConfig,
    (fused_attention, fused_key_value): (Option<&FusedAttention>, Option<&FusedKeyValue>),
    [batch, sequence]: [i32; 2],
    stream: &Stream,
) -> Result<Projected> {
    let shape = |heads| [batch, sequence, heads, config.head_dim];
    let (queries, keys, values) = match &weights.key_values {
        KeyValueWeights::Shared { step, source } => {
            let (keys, values) = step.key_values(*source)?;
            (weights.query.forward(input, stream)?, keys, values)
        },
        KeyValueWeights::Projected(projected) => {
            let ProjectedKeyValues { key, value, norm, publish } = projected.as_ref();
            let (queries, raw_keys, raw_values) = match (fused_attention, fused_key_value) {
                (Some(fused), _) => {
                    let output = fused.forward(input, stream)?;
                    (output.query, output.key, output.value)
                },
                (None, Some(fused)) => {
                    let (key, value) = fused.forward(input, stream)?;
                    (weights.query.forward(input, stream)?, key, Some(value))
                },
                (None, None) => (
                    weights.query.forward(input, stream)?,
                    key.forward(input, stream)?,
                    value.as_ref().map(|value| value.forward(input, stream)).transpose()?,
                ),
            };
            let raw_keys = raw_keys.reshape(&shape(config.kv_heads), stream)?;
            let values = match raw_values {
                _ if config.use_k_eq_v => raw_keys.rms_norm_unit(config.rms_norm_eps, stream)?,
                Some(values) => values
                    .reshape(&shape(config.kv_heads), stream)?
                    .rms_norm_unit(config.rms_norm_eps, stream)?,
                None => {
                    return Err(crate::engine::Error::InvalidModel(
                        "missing hybrid MoE value projection".into(),
                    ));
                },
            }
            .transpose(&[0, 2, 1, 3], stream)?;
            let keys = norm.apply(&raw_keys, config.rms_norm_eps, stream)?;
            if let Some(publish) = publish {
                publish.step.publish(publish.layer, &keys, &values)?;
            }
            (queries, keys, values)
        },
    };
    let queries = queries.reshape(&shape(config.attention_heads), stream)?;
    let queries = weights.query_norm.apply(&queries, config.rms_norm_eps, stream)?;
    Ok(Projected { queries, keys, values })
}

pub(super) fn rope_layout(
    input: &Array,
    frequencies: Option<&Array>,
    config: HybridMoeLayerConfig,
    position: i32,
    stream: &Stream,
) -> Result<Array> {
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    frequencies.map_or_else(
        || {
            input.rope(
                RopeOptions {
                    dimensions: config.rope_dimensions,
                    traditional: config.rope_traditional,
                    base: Some(config.rope_base),
                    scale: config.rope_position_scale,
                    offset: position,
                },
                stream,
            )
        },
        |frequencies| {
            input.rope_with_frequencies(
                config.rope_dimensions,
                config.rope_traditional,
                frequencies,
                position,
                stream,
            )
        },
    )
}
