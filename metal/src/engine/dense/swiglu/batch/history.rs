//! Decode attention over one persistent `[B, H, T, D]` history per layer.
//! Common-position rows rotate together, append in place and share one SDPA
//! call, so attention cost no longer grows with a dispatch per row.

use super::{
    super::{
        attention as dense_attention, config::DenseSwiGluLayerConfig, weights::AttentionWeights,
    },
    attention::row_slice,
};
use crate::engine::{
    Array, KvCache, PagedContextMode, Result, Stream, paged_attention_min_context,
    persistent_history,
};

/// Rotates `[B, S, H, D]` rows into `[B, H, S, D]`, each at its own position.
fn rotate(
    input: &Array,
    config: DenseSwiGluLayerConfig,
    positions: &[i32],
    stream: &Stream,
) -> Result<Array> {
    if let Some(position) =
        positions.first().copied().filter(|first| positions.iter().all(|p| p == first))
    {
        return dense_attention::rope_layout(input, None, config, position, stream);
    }
    let offsets = positions.iter().map(|p| Ok(u32::try_from(*p)?)).collect::<Result<Vec<_>>>()?;
    let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    Array::from_native(stream.native().graph().rope_batched(
        input.native(),
        mirtal::RopeOptions {
            dimensions: usize::try_from(config.head_dim)?,
            traditional: false,
            base: Some(config.rope_base),
            scale: 1.0,
            offset: &offsets,
        },
    )?)
}

/// Returns `None` before touching any cache when per-row attention must run.
#[allow(clippy::too_many_arguments)]
pub(super) fn persistent_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    caches: &mut [&mut KvCache],
    weights: &AttentionWeights,
    config: DenseSwiGluLayerConfig,
    positions: &[i32],
    causal: bool,
    stream: &Stream,
) -> Result<Option<Array>> {
    let selected = stream.config().batch.dense_history.persistent()
        && !causal
        && caches.len() > 1
        && weights.rope_frequencies.is_none()
        && config.rope_attention_factor.to_bits() == 1.0_f32.to_bits()
        && queries.shape()?[1] == 1
        && caches.iter().all(|cache| !cache.has_quantized_pages());
    if !selected {
        return Ok(None);
    }
    let layout =
        [caches.len(), usize::try_from(config.kv_heads)?, usize::try_from(config.head_dim)?];
    let tokens = positions
        .iter()
        .map(|position| Ok(usize::try_from(*position)? + 1))
        .collect::<Result<Vec<_>>>()?;
    let previous = persistent_history::take(caches);
    if !persistent_history::admits(
        previous.as_deref(),
        layout,
        &tokens,
        keys.native().dtype()?,
        stream,
    )? {
        return Ok(None);
    }
    let queries = rotate(queries, config, positions, stream)?;
    let keys = rotate(keys, config, positions, stream)?;
    let contexts = caches
        .iter_mut()
        .enumerate()
        .map(|(row, cache)| {
            cache.update_for_attention_mode(
                &row_slice(&keys, row, config.kv_heads, config.head_dim, stream)?,
                &row_slice(values, row, config.kv_heads, config.head_dim, stream)?,
                stream,
                paged_attention_min_context(stream),
                PagedContextMode::View,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let scale = config.attention_scale;
    if persistent_history::enabled(caches, &contexts, false, 1) {
        return persistent_history::execute(
            previous.as_deref(),
            caches,
            &contexts,
            [&keys, values],
            &queries,
            scale,
            stream,
        )
        .map(Some);
    }
    let rows = contexts
        .iter()
        .enumerate()
        .map(|(row, context)| {
            row_slice(&queries, row, config.heads, config.head_dim, stream)?
                .scaled_dot_product_attention(&context.keys, &context.values, scale, false, stream)
        })
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream).map(Some)
}
