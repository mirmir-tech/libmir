use super::{config::DenseSwiGluLayerConfig, weights::AttentionWeights};
use crate::engine::{
    Array, FusedAttention, KvCache, NormWeight, PagedContextMode, Result, RopeOptions, Stream,
    native_paged_attention_mode, paged_attention_min_context,
};

pub(super) struct AttentionContext<'a> {
    pub cache: &'a mut KvCache,
    pub position: i32,
    pub causal: bool,
    pub stream: &'a Stream,
}

pub(super) fn forward(
    input: &Array,
    weights: &AttentionWeights,
    fused_attention: Option<&FusedAttention>,
    config: DenseSwiGluLayerConfig,
    context: AttentionContext<'_>,
) -> Result<Array> {
    let AttentionContext { cache, position, causal, stream } = context;
    let sequence = input.shape()?.get(1).copied().ok_or_else(|| {
        crate::engine::Error::InvalidModel(
            "dense SwiGLU attention input has no sequence axis".into(),
        )
    })?;
    let fused = (sequence == 1)
        .then_some(fused_attention)
        .flatten()
        .map(|fused| fused.forward(input, stream))
        .transpose()?;
    let (queries, keys, values) = match fused {
        Some(output) => (
            output.query,
            output.key,
            output.value.ok_or_else(|| {
                crate::engine::Error::InvalidModel("fused attention omitted values".into())
            })?,
        ),
        None => (
            weights.query.forward(input, stream)?,
            weights.key.forward(input, stream)?,
            weights.value.forward(input, stream)?,
        ),
    };
    let queries = queries.reshape(&[1, sequence, config.heads, config.head_dim], stream)?;
    let queries = normalize(queries, weights.query_norm.as_ref(), config.rms_norm_eps, stream)?;
    let queries =
        rope_layout(&queries, weights.rope_frequencies.as_ref(), config, position, stream)?;
    let keys = keys.reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?;
    let keys = normalize(keys, weights.key_norm.as_ref(), config.rms_norm_eps, stream)?;
    let keys = rope_layout(&keys, weights.rope_frequencies.as_ref(), config, position, stream)?;
    let values = values
        .reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    let mode = if sequence == 1 {
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
        &keys,
        &values,
        stream,
        paged_attention_min_context(stream),
        mode,
    )?;
    let output = match context.paged {
        Some(paged) => queries.paged_scaled_dot_product_attention_with_scratch(
            paged.attention(),
            paged.scratch(),
            config.attention_scale,
            stream,
        )?,
        None => queries.scaled_dot_product_attention(
            &context.keys,
            &context.values,
            config.attention_scale,
            causal,
            stream,
        )?,
    };
    let output = output.transpose(&[0, 2, 1, 3], stream)?;
    let output_width = config.heads * config.head_dim;
    weights
        .output
        .forward(&output.reshape(&[1, sequence, output_width], stream)?, stream)
}

pub(super) fn normalize(
    input: Array,
    weight: Option<&NormWeight>,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    match weight {
        Some(weight) => weight.apply(&input, eps, stream),
        None => Ok(input),
    }
}

pub(super) fn rope_layout(
    input: &Array,
    frequencies: Option<&Array>,
    config: DenseSwiGluLayerConfig,
    position: i32,
    stream: &Stream,
) -> Result<Array> {
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    let rotated = frequencies.map_or_else(
        || {
            input.rope(
                RopeOptions {
                    dimensions: config.head_dim,
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
    )?;
    if config.rope_attention_factor.to_bits() == 1.0_f32.to_bits() {
        Ok(rotated)
    } else {
        rotated.multiply_scalar(config.rope_attention_factor, stream)
    }
}
