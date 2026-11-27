use std::sync::Arc;

use super::{
    HybridMoeLayerConfig,
    sharing::SourceAttention,
    weights::{AttentionWeights, KeyValueWeights, ProjectedKeyValues, Publish},
};
use crate::engine::{
    Array, Error, FusedAttention, FusedKeyValue, KvCache, KvContext, PagedContextMode, Result,
    RopeOptions, Stream, native_paged_attention_mode, paged_attention_min_context,
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
    let sequence = input
        .shape()?
        .get(1)
        .copied()
        .ok_or_else(|| Error::InvalidModel("attention input has no sequence axis".into()))?;
    let fused = if sequence == 1 {
        (fused_attention, fused_key_value)
    } else {
        (None, None)
    };
    let Projected { queries, key_values } =
        project(input, weights, config, fused, [1, sequence], stream)?;
    let rotate = |input: &Array| {
        rope_layout(input, weights.rope_frequencies.as_ref(), config, position, stream)
    };
    let queries = rotate(&queries)?;
    let output = match key_values {
        KeyValues::Shared(attended) => {
            let SourceAttention::Rows(rows) = attended.as_ref() else {
                return Err(Error::InvalidModel(
                    "a lone sequence shares no persistent history".into(),
                ));
            };
            let [(context, _)] = rows.as_slice() else {
                return Err(Error::InvalidModel(
                    "the shared attention is not one sequence's".into(),
                ));
            };
            attend(&queries, context, mask, causal, stream)?
        },
        KeyValues::Own { keys, values, publish } => {
            let keys = rotate(&keys)?;
            let (context, mode) = match cache {
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
                        PagedContextMode::View
                    };
                    let context = cache.update_for_attention_mode(
                        &keys,
                        &values,
                        stream,
                        paged_attention_min_context(stream),
                        mode,
                    )?;
                    (context, mode)
                },
                None => {
                    (KvContext { keys, values, paged: None, mask: None }, PagedContextMode::View)
                },
            };
            if let Some(publish) = publish {
                publish.publish(SourceAttention::Rows(vec![(context.share()?, mode)]))?;
            }
            attend(&queries, &context, mask, causal, stream)?
        },
    };
    let output = output.transpose(&[0, 2, 1, 3], stream)?;
    let output_width = config.attention_heads * config.head_dim;
    let output = output.reshape(&[1, sequence, output_width], stream)?;
    weights.output.forward(&output, stream)
}

/// Rotated `[1, heads, S, D]` queries over `context`: the paged one a
/// decode reads natively, else with the context's or the layer's mask. A
/// multi-token pass always asks its cache for a view, so a paged context
/// there is a broken invariant, refused rather than read.
fn attend(
    queries: &Array,
    context: &KvContext,
    mask: Option<&Array>,
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    context.paged.as_ref().map_or_else(
        || {
            mask.or(context.mask.as_ref()).map_or_else(
                || {
                    queries.scaled_dot_product_attention(
                        &context.keys, &context.values, 1.0, causal, stream,
                    )
                },
                |mask| {
                    queries.masked_scaled_dot_product_attention(
                        &context.keys, &context.values, 1.0, mask, stream,
                    )
                },
            )
        },
        |paged| {
            if queries.shape()?.get(2).is_none_or(|&tokens| tokens != 1) {
                return Err(Error::MultiTokenPagedAttention);
            }
            queries.paged_scaled_dot_product_attention_with_scratch(
                paged.attention(),
                paged.scratch(),
                1.0,
                stream,
            )
        },
    )
}

/// Normalized, unrotated `[B, S, heads, D]` queries, and the keys and values
/// they attend over.
pub(super) struct Projected<'a> {
    pub(super) queries: Array,
    pub(super) key_values: KeyValues<'a>,
}

pub(super) enum KeyValues<'a> {
    /// Unrotated `[B, S, kv_heads, D]` keys and `[B, kv_heads, S, D]`
    /// values, and the slot a source layer publishes what it attends over to.
    Own {
        keys: Array,
        values: Array,
        publish: Option<&'a Publish>,
    },
    /// What the source layer attended over in the current step.
    Shared(Arc<SourceAttention>),
}

/// The layer's queries, and its own keys and values or what the earlier
/// layer it shares them with attended over in the same step.
pub(super) fn project<'a>(
    input: &Array,
    weights: &'a AttentionWeights,
    config: HybridMoeLayerConfig,
    (fused_attention, fused_key_value): (Option<&FusedAttention>, Option<&FusedKeyValue>),
    [batch, sequence]: [i32; 2],
    stream: &Stream,
) -> Result<Projected<'a>> {
    let shape = |heads| [batch, sequence, heads, config.head_dim];
    let (queries, key_values) = match &weights.key_values {
        KeyValueWeights::Shared { step, source } => (
            weights.query.forward(input, stream)?,
            KeyValues::Shared(step.attended(*source)?),
        ),
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
                    return Err(Error::InvalidModel("missing hybrid MoE value projection".into()));
                },
            }
            .transpose(&[0, 2, 1, 3], stream)?;
            let keys = norm.apply(&raw_keys, config.rms_norm_eps, stream)?;
            (queries, KeyValues::Own { keys, values, publish: publish.as_ref() })
        },
    };
    let queries = queries.reshape(&shape(config.attention_heads), stream)?;
    let queries = weights.query_norm.apply(&queries, config.rms_norm_eps, stream)?;
    Ok(Projected { queries, key_values })
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
