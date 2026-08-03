use super::{HybridMoeLayerConfig, weights::AttentionWeights};
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
    let fused_attention = (sequence == 1)
        .then_some(fused_attention)
        .flatten()
        .map(|fused| fused.forward(input, stream))
        .transpose()?;
    let (queries, raw_keys, raw_values) = if let Some(output) = fused_attention {
        (output.query, output.key, output.value)
    } else {
        let fused_key_value = (sequence == 1)
            .then_some(fused_key_value)
            .flatten()
            .map(|fused| fused.forward(input, stream))
            .transpose()?;
        let (raw_keys, raw_values) = match fused_key_value {
            Some((key, value)) => (key, Some(value)),
            None => (
                weights.key.forward(input, stream)?,
                weights.value.as_ref().map(|value| value.forward(input, stream)).transpose()?,
            ),
        };
        (weights.query.forward(input, stream)?, raw_keys, raw_values)
    };
    let queries =
        queries.reshape(&[1, sequence, config.attention_heads, config.head_dim], stream)?;
    let queries = weights.query_norm.apply(&queries, config.rms_norm_eps, stream)?;
    let queries =
        rope_layout(&queries, weights.rope_frequencies.as_ref(), config, position, stream)?;

    let raw_keys = raw_keys.reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?;
    let values = if config.use_k_eq_v {
        raw_keys
            .rms_norm_unit(config.rms_norm_eps, stream)?
            .transpose(&[0, 2, 1, 3], stream)?
    } else {
        let raw_values = raw_values.ok_or_else(|| {
            crate::engine::Error::InvalidModel("missing hybrid MoE value projection".into())
        })?;
        raw_values
            .reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?
            .rms_norm_unit(config.rms_norm_eps, stream)?
            .transpose(&[0, 2, 1, 3], stream)?
    };
    let keys = weights.key_norm.apply(&raw_keys, config.rms_norm_eps, stream)?;
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
                    traditional: false,
                    base: Some(config.rope_base),
                    scale: 1.0,
                    offset: position,
                },
                stream,
            )
        },
        |frequencies| {
            input.rope_with_frequencies(config.head_dim, false, frequencies, position, stream)
        },
    )
}
