//! Attention whose keys or query scale depend on the query's position
//! (Llama 4): chunked attention, where a query sees only the keys since the
//! start of its `chunk`-aligned block, and a query temperature on the
//! unrotated layers. Chunked layers keep a window cache of one chunk, whose
//! view may be the decode ring; packed rows run one at a time.

use mirtal::DType;

use super::{
    attention::{self, AttentionContext},
    config::DenseSwiGluLayerConfig,
    weights::AttentionWeights,
};
use crate::engine::{Array, KvCache, KvContext, Result, Stream};

/// Queries `[1, heads, L, D]` at positions from `position`, scaled by the
/// layer's temperature where it is not one.
pub(super) fn temper(
    queries: Array,
    config: DenseSwiGluLayerConfig,
    position: usize,
    stream: &Stream,
) -> Result<Array> {
    let Some(temperature) = config.temperature else {
        return Ok(queries);
    };
    let sequence = usize::try_from(queries.shape()?[2])?;
    if position + sequence <= temperature.first_scaled_position() {
        return Ok(queries);
    }
    let factors = (position..position + sequence)
        .map(|position| Ok(temperature.factor(position).to_string().parse::<f32>()?))
        .collect::<Result<Vec<_>>>()?;
    let factors = Array::from_f32(&factors, &[1, 1, i32::try_from(sequence)?, 1])?;
    queries.multiply(&factors.astype_like(&queries, stream)?, stream)
}

/// Where a K/V view's tokens sit: slot `newest` holds position `last`, and
/// each earlier slot, cyclically, the position before — a chronological view
/// and a window cache's decode ring alike.
#[derive(Debug, Clone, Copy)]
pub(super) struct ViewOrigin {
    pub(super) newest: usize,
    pub(super) last: usize,
}

/// Chunked attention of `queries [1, heads, L, D]` at the last `L`
/// positions up to `origin.last`, once they reach past the first chunk;
/// `None` while every query still sees every earlier key.
pub(super) fn attend(
    queries: &Array,
    context: &KvContext,
    config: DenseSwiGluLayerConfig,
    origin: ViewOrigin,
    stream: &Stream,
) -> Result<Option<Array>> {
    let Some((window, chunk)) = config.window.zip(config.chunk()) else {
        return Ok(None);
    };
    let slots = usize::try_from(context.keys.shape()?[2])?;
    let sequence = usize::try_from(queries.shape()?[2])?;
    let last = origin.last;
    let start = window.first_visible(last);
    if start == 0 {
        return Ok(None);
    }
    let first = last + 1 - sequence;
    // A lone query attends to its chunk's slots directly when they are one
    // run of the view: the newest `age + 1`, unless the ring wraps them.
    let age = last - start;
    if sequence == 1 && (age + 1 >= slots || origin.newest >= age) {
        let (begin, end) = if age + 1 >= slots {
            (0, slots)
        } else {
            (origin.newest - age, origin.newest + 1)
        };
        return queries
            .scaled_dot_product_attention(
                &slots_between(&context.keys, begin, end, stream)?,
                &slots_between(&context.values, begin, end, stream)?,
                config.attention_scale,
                false,
                stream,
            )
            .map(Some);
    }
    let mask = chunk_mask(origin, slots, first, chunk, stream)?;
    queries
        .masked_scaled_dot_product_attention(
            &context.keys,
            &context.values,
            config.attention_scale,
            &mask,
            stream,
        )
        .map(Some)
}

/// Slots `begin..end` of a `[1, heads, slots, D]` view.
fn slots_between(view: &Array, begin: usize, end: usize, stream: &Stream) -> Result<Array> {
    let shape = view
        .shape()?
        .iter()
        .map(|&dim| usize::try_from(dim))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut start = vec![0; shape.len()];
    let mut stop = shape;
    start[2] = begin;
    stop[2] = end;
    view.slice(&start, &stop, stream)
}

/// `[L, slots]` keys each query at positions `first..=origin.last` may see:
/// earlier or itself, and in its own chunk. Slot `i` holds position
/// `last - ((newest - i) mod slots)`.
fn chunk_mask(
    origin: ViewOrigin,
    slots: usize,
    first: usize,
    chunk: usize,
    stream: &Stream,
) -> Result<Array> {
    let graph = stream.native().graph();
    let float = |value: usize| -> Result<f32> { Ok(value.to_string().parse()?) };
    let scalar = |value: usize| -> Result<mirtal::Array> {
        Ok(graph.full(&mirtal::Shape::new([])?, float(value)?, DType::Int32)?)
    };
    let slot = graph.arange(0.0, float(slots)?, 1.0, DType::Int32)?;
    let newest = scalar(origin.newest)?;
    // (newest - i) mod slots, as newest - i plus slots where i is past it.
    let wrapped = graph.astype(&graph.less(&newest, &slot)?, DType::Int32)?;
    let age =
        graph.add(&graph.subtract(&newest, &slot)?, &graph.multiply(&wrapped, &scalar(slots)?)?)?;
    let keys = graph.expand_dims(&graph.subtract(&scalar(origin.last)?, &age)?, &[0])?;
    let queries = graph.expand_dims(
        &graph.arange(float(first)?, float(origin.last + 1)?, 1.0, DType::Int32)?,
        &[-1],
    )?;
    let size = scalar(chunk)?;
    let key_chunk = graph.floor_divide(&keys, &size)?;
    let query_chunk = graph.floor_divide(&queries, &size)?;
    let same_chunk = graph.logical_and(
        &graph.greater_equal(&key_chunk, &query_chunk)?,
        &graph.greater_equal(&query_chunk, &key_chunk)?,
    )?;
    Array::from_native(graph.logical_and(&graph.greater_equal(&queries, &keys)?, &same_chunk)?)
}

/// Packed rows through the per-row path.
#[allow(clippy::too_many_arguments)]
pub(super) fn packed(
    input: &Array,
    weights: &AttentionWeights,
    config: DenseSwiGluLayerConfig,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape()?;
    let (sequence, hidden) = (usize::try_from(shape[1])?, usize::try_from(shape[2])?);
    let rows = caches
        .iter_mut()
        .enumerate()
        .map(|(row, cache)| {
            let input = input.slice(&[row, 0, 0], &[row + 1, sequence, hidden], stream)?;
            attention::forward(
                &input,
                weights,
                None,
                config,
                AttentionContext {
                    cache,
                    position: positions[row],
                    causal,
                    stream,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)
}

#[cfg(test)]
mod tests;
