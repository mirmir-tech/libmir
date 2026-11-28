//! Decode attention over one persistent `[B, H, T, D]` history per layer.
//! Rows rotate together, append in place and share one SDPA call, so attention
//! cost no longer grows with a dispatch per row.

use std::sync::Arc;

use super::{
    Attended, History, KvCache, admits, attend_history, enabled, execute_publishing, row_attention,
    take,
};
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
    Ok(attend_publishing(layer, caches, positions, rotate, stream)?.map(|(output, _)| output))
}

/// [`attend`], also returning what the rows attended over, for later layers
/// that attend over the same keys and values.
pub fn attend_publishing(
    layer: &Attend<'_>,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    rotate: impl Fn(&Array) -> Result<Array>,
    stream: &Stream,
) -> Result<Option<(Array, Attended)>> {
    let shape = layer.keys.shape()?;
    let &[_, sequence, kv_heads, head_dim] = shape.as_slice() else {
        return Ok(None);
    };
    let selected = stream.config().batch.dense_history.persistent()
        && sequence == 1
        && caches.iter().all(|cache| !cache.is_windowed() && !cache.has_quantized_pages());
    // A lone session gains only once fragmented pages would need a gather; below
    // the paging threshold its contiguous K/V stays the single attention input.
    if !selected || (caches.len() == 1 && !caches[0].has_active_pages()) {
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
    // A reused history already holds every earlier token: one page write for
    // all rows replaces a write and a gathered view per row.
    if let Some(previous) = previous.as_deref()
        && previous.extends_to(caches.len(), kv_heads, &tokens, head_dim)
        && previous.dtype_matches(dtype)?
        && KvCache::write_rows(caches, [&keys, layer.values], stream)?
    {
        let history = Arc::new(History::appended(previous, [&keys, layer.values], stream)?);
        let output = attend_history(&history, caches, &queries, layer.scale, layer.sinks, stream)?;
        return Ok(Some((output, Attended::History(history))));
    }
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
        let (output, history) = execute_publishing(
            previous.as_deref(),
            caches,
            &contexts,
            [&keys, layer.values],
            &queries,
            layer.scale,
            layer.sinks,
            stream,
        )?;
        let attended = history.map_or_else(|| Attended::Rows(contexts), Attended::History);
        return Ok(Some((output, attended)));
    }
    let output = row_attention(&queries, &contexts, layer.scale, layer.sinks, stream)?;
    Ok(Some((output, Attended::Rows(contexts))))
}
