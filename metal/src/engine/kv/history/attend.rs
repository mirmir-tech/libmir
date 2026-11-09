//! Decode attention over one persistent `[B, H, T, D]` history per layer.
//! Rows rotate together, append in place and share one SDPA call, so attention
//! cost no longer grows with a dispatch per row.

use super::{KvCache, admits, enabled, execute, row_attention, take};
use crate::engine::{Array, PagedContextMode, Result, Stream, paged_attention_min_context};

pub struct Attend<'a> {
    /// `[B, 1, H, D]`, not yet rotated.
    pub queries: &'a Array,
    /// `[B, 1, H, D]`, not yet rotated.
    pub keys: &'a Array,
    /// `[B, H, 1, D]`.
    pub values: &'a Array,
    pub scale: f32,
    pub sinks: Option<&'a Array>,
}

/// Returns `None` before touching any cache when per-row attention must run.
/// `rotate` maps queries and keys to `[B, H, 1, D]` at the row positions.
pub fn attend(
    layer: &Attend<'_>,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    rotate: impl Fn(&Array) -> Result<Array>,
    stream: &Stream,
) -> Result<Option<Array>> {
    let shape = layer.keys.shape()?;
    let &[_, sequence, kv_heads, head_dim] = shape.as_slice() else {
        return Ok(None);
    };
    let selected = stream.config().batch.dense_history.persistent()
        && sequence == 1
        && caches.len() > 1
        && caches
            .iter()
            .all(|cache| cache.max_context.is_none() && !cache.has_quantized_pages());
    if !selected {
        return Ok(None);
    }
    let (kv_heads, head_dim) = (usize::try_from(kv_heads)?, usize::try_from(head_dim)?);
    let tokens = positions
        .iter()
        .map(|position| Ok(usize::try_from(*position)? + 1))
        .collect::<Result<Vec<_>>>()?;
    let previous = take(caches);
    let layout = [caches.len(), kv_heads, head_dim];
    let dtype = layer.keys.native().dtype()?;
    if !admits(previous.as_deref(), layout, &tokens, dtype, stream)? {
        return Ok(None);
    }
    let queries = rotate(layer.queries)?;
    let keys = rotate(layer.keys)?;
    let row = |array: &Array, index: usize| {
        array.slice(&[index, 0, 0, 0], &[index + 1, kv_heads, 1, head_dim], stream)
    };
    let contexts = caches
        .iter_mut()
        .enumerate()
        .map(|(index, cache)| {
            cache.update_for_attention_mode(
                &row(&keys, index)?,
                &row(layer.values, index)?,
                stream,
                paged_attention_min_context(stream),
                PagedContextMode::View,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    if enabled(caches, &contexts, false, 1) {
        return execute(
            previous.as_deref(),
            caches,
            &contexts,
            [&keys, layer.values],
            &queries,
            layer.scale,
            layer.sinks,
            stream,
        )
        .map(Some);
    }
    row_attention(&queries, &contexts, layer.scale, layer.sinks, stream).map(Some)
}
