//! Softmax attention with a score cap, `cap · tanh(s / cap)` before the
//! softmax (Gemma 2's `attn_logit_softcapping`), which fused scaled
//! dot-product attention cannot apply: scores form explicitly over the
//! chronological K/V view.

use mirtal::{DType, Shape};

use super::config::DenseSwiGluLayerConfig;
use crate::engine::{Array, KvContext, Result, Stream};

/// Score shift that removes a masked key from the softmax.
const MASKED: f32 = -1e30;

/// `queries [1, heads, L, D]` over the context's `[1, kv_heads, T, D]` keys
/// and values, the last `L` keys being the queries' own positions.
pub(super) fn attend(
    queries: &Array,
    context: &KvContext,
    config: DenseSwiGluLayerConfig,
    cap: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    let graph = stream.native().graph();
    let shape = queries.shape()?;
    let (heads, sequence, width) = (shape[1], shape[2], shape[3]);
    let total = context.keys.shape()?[2];
    let groups = heads / config.kv_heads;
    let shape5 = |dims: [i32; 5]| Shape::new(dims.map(|dim| usize::try_from(dim).unwrap_or(0)));
    let query = graph.reshape(
        &graph.astype(queries.native(), DType::Float32)?,
        &shape5([1, config.kv_heads, groups, sequence, width])?,
    )?;
    let keys = graph.reshape(
        &graph.astype(context.keys.native(), DType::Float32)?,
        &shape5([1, config.kv_heads, 1, total, width])?,
    )?;
    let values = graph.reshape(
        &graph.astype(context.values.native(), DType::Float32)?,
        &shape5([1, config.kv_heads, 1, total, width])?,
    )?;
    let scores = graph.multiply_scalar(
        &graph.matmul(&query, &graph.transpose(&keys, &[0, 1, 2, 4, 3])?)?,
        config.attention_scale,
    )?;
    let scores = Array::from_native(scores)?.logit_softcap(cap, stream)?;
    let scores = match (context.mask.as_ref(), causal && sequence > 1) {
        (Some(mask), _) => scores.add(&mask.astype_like(&scores, stream)?, stream)?,
        (None, true) => scores.add(&future_mask(sequence, total, stream)?, stream)?,
        (None, false) => scores,
    };
    let probabilities = graph.softmax(scores.native(), -1, true)?;
    let output = graph.reshape(
        &graph.matmul(&probabilities, &values)?,
        &Shape::new([1, heads, sequence, width].map(|dim| usize::try_from(dim).unwrap_or(0)))?,
    )?;
    Array::from_native(output)?.astype_like(queries, stream)
}

/// `[L, T]` additive mask hiding keys after each query, the queries being
/// the last `L` of `T` positions: `clip(key - query, 0, 1) · MASKED`.
fn future_mask(sequence: i32, total: i32, stream: &Stream) -> Result<Array> {
    let graph = stream.native().graph();
    let first = f32::from(u16::try_from(total - sequence)?);
    let queries = graph.reshape(
        &graph.arange(first, first + f32::from(u16::try_from(sequence)?), 1.0, DType::Float32)?,
        &Shape::new([usize::try_from(sequence)?, 1])?,
    )?;
    let keys = graph.reshape(
        &graph.arange(0.0, f32::from(u16::try_from(total)?), 1.0, DType::Float32)?,
        &Shape::new([1, usize::try_from(total)?])?,
    )?;
    let scalar = |value: f32| graph.full(&Shape::new([])?, value, DType::Float32);
    let future = graph.clip(
        &graph.add(&keys, &graph.multiply_scalar(&queries, -1.0)?)?,
        &scalar(0.0)?,
        &scalar(1.0)?,
    )?;
    Array::from_native(graph.multiply_scalar(&future, MASKED)?)
}

#[cfg(test)]
mod tests;
