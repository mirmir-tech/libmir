mod rope;

pub(in crate::engine::dense::swiglu) use rope::rope_layout;

use super::{
    config::DenseSwiGluLayerConfig,
    weights::{AttentionWeights, QkNorm, QkSide},
};
use crate::engine::{
    Array, FusedAttention, KvCache, NormWeight, PagedContextMode, Result, Stream,
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
    let queries =
        heads(queries, QkSide::Query, weights, config, [1, sequence, config.heads], stream)?;
    let keys = heads(keys, QkSide::Key, weights, config, [1, sequence, config.kv_heads], stream)?;
    let values = values
        .reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    let output_width = config.heads * config.head_dim;
    if let Some(attended) = super::batch::persistent_attention(
        &queries,
        &keys,
        &values,
        &mut [&mut *cache],
        weights,
        config,
        &[position],
        causal,
        stream,
    )? {
        return weights.output.forward(
            &attended
                .transpose(&[0, 2, 1, 3], stream)?
                .reshape(&[1, sequence, output_width], stream)?,
            stream,
        );
    }
    let queries =
        rope_layout(&queries, weights.rope_frequencies.as_ref(), config, position, stream)?;
    let queries = super::positional::temper(queries, config, usize::try_from(position)?, stream)?;
    let keys = rope_layout(&keys, weights.rope_frequencies.as_ref(), config, position, stream)?;
    // Capped scores and chunks form over the chronological K/V view.
    let mode = if sequence == 1 && config.attention_softcap.is_none() && config.chunk().is_none() {
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
    let origin = super::positional::ViewOrigin {
        newest: match cache.window_newest_slot() {
            Some(slot) => slot,
            None => usize::try_from(context.keys.shape()?[2])? - 1,
        },
        last: usize::try_from(position)? + usize::try_from(sequence)? - 1,
    };
    let output = match super::positional::attend(&queries, &context, config, origin, stream)? {
        Some(output) => output,
        None => attend(&queries, &context, config, causal, stream)?,
    };
    let output = output.transpose(&[0, 2, 1, 3], stream)?;
    weights
        .output
        .forward(&output.reshape(&[1, sequence, output_width], stream)?, stream)
}

/// Attention over the updated context: paged kernels, the chronological
/// view with its mask, or capped scores.
fn attend(
    queries: &Array,
    context: &crate::engine::KvContext,
    config: DenseSwiGluLayerConfig,
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    Ok(match (context.paged.as_ref(), config.attention_softcap) {
        (_, Some(cap)) => super::softcap::attend(queries, context, config, cap, causal, stream)?,
        (Some(paged), None) => queries.paged_scaled_dot_product_attention_with_scratch(
            paged.attention(),
            paged.scratch(),
            config.attention_scale,
            stream,
        )?,
        // A sliding cache's bulk write carries its causal window mask.
        (None, None) => match context.mask.as_ref() {
            Some(mask) => queries.masked_scaled_dot_product_attention(
                &context.keys,
                &context.values,
                config.attention_scale,
                mask,
                stream,
            )?,
            None => queries.scaled_dot_product_attention(
                &context.keys,
                &context.values,
                config.attention_scale,
                causal,
                stream,
            )?,
        },
    })
}

/// Projections `[rows, sequence, heads · head_dim]` as `[rows, sequence,
/// heads, head_dim]`, normalized per head or, for a norm across heads,
/// before the split.
pub(super) fn heads(
    projected: Array,
    side: QkSide,
    weights: &AttentionWeights,
    config: DenseSwiGluLayerConfig,
    [rows, sequence, heads]: [i32; 3],
    stream: &Stream,
) -> Result<Array> {
    let shape = [rows, sequence, heads, config.head_dim];
    let norm = weights.qk_norm.weight(side);
    match weights.qk_norm {
        QkNorm::None => projected.reshape(&shape, stream),
        QkNorm::Projection { .. } => {
            normalize(projected, norm, config.rms_norm_eps, stream)?.reshape(&shape, stream)
        },
        QkNorm::Head { .. } => {
            normalize(projected.reshape(&shape, stream)?, norm, config.rms_norm_eps, stream)
        },
        // transformers `Llama4TextL2Norm(config.rms_norm_eps)`, as llama.cpp.
        QkNorm::UnitHead => {
            projected.reshape(&shape, stream)?.rms_norm_unit(config.rms_norm_eps, stream)
        },
    }
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
