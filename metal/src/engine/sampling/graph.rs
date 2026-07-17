use mirtal::{Array as NativeArray, DType, Error as NativeError, Graph, Shape};

use super::DeviceSampling;
use crate::engine::Array;

#[derive(Debug)]
pub struct TopK {
    pub token_ids: Array,
    pub scores: Array,
}

pub(super) fn argmax(graph: Graph<'_>, input: &NativeArray) -> mirtal::Result<NativeArray> {
    graph.argmax_axis(input, -1, false)
}

pub(super) fn top_k(
    graph: Graph<'_>,
    input: &NativeArray,
    k: usize,
    vocab_size: usize,
) -> mirtal::Result<[NativeArray; 2]> {
    let logits = vocab_logits(graph, input, vocab_size)?;
    top_candidates(graph, &logits, k)
}

pub(super) fn sample(
    graph: Graph<'_>,
    input: &NativeArray,
    sampling: DeviceSampling,
) -> mirtal::Result<NativeArray> {
    let logits = vocab_logits(graph, input, sampling.vocab_size)?;
    let [mut indices, mut scores] = top_candidates(graph, &logits, sampling.top_k)?;
    let order = graph.argsort(&graph.negative(&scores)?, -1)?;
    indices = graph.take_along_axis(&indices, &order, -1)?;
    scores = graph.take_along_axis(&scores, &order, -1)?;
    let total = graph.logsumexp(&logits, -1, true)?;
    let nucleus = graph.exp(&graph.subtract(&scores, &total)?)?;
    let prior = graph.cumulative_sum(&nucleus, -1, false, false)?;
    let keep = graph.less(&prior, &scalar(graph, sampling.top_p, scores.dtype()?)?)?;
    let maximum = graph.reduce_max(&scores, -1, true)?;
    let centered = graph.subtract(&scores, &maximum)?;
    let temperature = scalar(graph, sampling.temperature, scores.dtype()?)?;
    let weights = graph.exp(&graph.divide(&centered, &temperature)?)?;
    let keep = graph.astype(&keep, scores.dtype()?)?;
    let weights = graph.multiply(&weights, &keep)?;
    let cumulative = graph.cumulative_sum(&weights, -1, false, true)?;
    let total = graph.reduce_sum(&weights, -1, true)?;
    let threshold = graph.multiply(&total, &scalar(graph, sampling.draw, scores.dtype()?)?)?;
    let position = graph.argmax_axis(&graph.greater_equal(&cumulative, &threshold)?, -1, false)?;
    let position = graph.expand_dims(&position, &[-1])?;
    let token = graph.take_along_axis(&indices, &position, -1)?;
    graph.squeeze_axis(&token, -1)
}

fn vocab_logits(
    graph: Graph<'_>,
    input: &NativeArray,
    vocab_size: usize,
) -> mirtal::Result<NativeArray> {
    let shape = input.shape()?;
    let available = shape
        .dimensions()
        .last()
        .copied()
        .ok_or_else(|| invalid("sampling logits must have rank"))?;
    if vocab_size == 0 || vocab_size > available {
        return Err(invalid("sampling vocabulary is outside logits range"));
    }
    if vocab_size == available {
        return Ok(input.clone());
    }
    let start = vec![0; shape.dimensions().len()];
    let mut stop = shape.dimensions().to_vec();
    let last = stop.len() - 1;
    stop[last] = vocab_size;
    graph.slice(input, &start, &stop)
}

fn top_candidates(
    graph: Graph<'_>,
    logits: &NativeArray,
    k: usize,
) -> mirtal::Result<[NativeArray; 2]> {
    let shape = logits.shape()?;
    let vocab = shape
        .dimensions()
        .last()
        .copied()
        .ok_or_else(|| invalid("missing vocab axis"))?;
    if k == 0 || k > vocab {
        return Err(invalid("sampling top-k is outside logits range"));
    }
    let indices = graph.argpartition(logits, i32::try_from(vocab - k)?, -1)?;
    let mut start = vec![0; shape.dimensions().len()];
    let stop = shape.dimensions().to_vec();
    let last = start.len() - 1;
    start[last] = vocab - k;
    let indices = graph.slice(&indices, &start, &stop)?;
    let scores = graph.take_along_axis(logits, &indices, -1)?;
    Ok([graph.astype(&indices, DType::Uint32)?, scores])
}

fn scalar(graph: Graph<'_>, value: f32, dtype: DType) -> mirtal::Result<NativeArray> {
    graph.full(&Shape::new([])?, value, dtype)
}

fn invalid(message: impl Into<String>) -> NativeError {
    NativeError::InvalidOperation(message.into())
}
