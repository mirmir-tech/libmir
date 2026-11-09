//! Mamba-2 SSM update: the batched per-token recurrence for one token and
//! the structured state-space duality chunks for several.

use mirtal::{DType, Graph, Shape};

use super::{MambaShape, StateSpaceState, convolution::split_rows};
use crate::engine::{Array, Result, Stream, gated_delta::StateArray};

#[derive(Debug)]
pub(super) struct RecurrenceWeights {
    pub(super) decay_log: Array,
    pub(super) skip_scale: Array,
    pub(super) time_bias: Array,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward(
    weights: &RecurrenceWeights,
    shape: MambaShape,
    x: &Array,
    b: &Array,
    c: &Array,
    raw_dt: &Array,
    states: &mut [&mut StateSpaceState],
    stream: &Stream,
) -> Result<Array> {
    let dimensions = x.shape()?;
    let (rows, sequence) = (usize::try_from(dimensions[0])?, usize::try_from(dimensions[1])?);
    let graph = stream.native().graph();
    let reshape = |array: &Array, dims: &[usize]| -> Result<mirtal::Array> {
        Ok(graph.reshape(array.native(), &Shape::new(dims.to_vec())?)?)
    };
    let (heads, head_dim, groups, state_size) =
        (shape.heads, shape.head_dim, shape.groups, shape.state_size);
    let x = reshape(x, &[rows, sequence, heads, head_dim])?;
    let b = reshape(b, &[rows, sequence, groups, state_size])?;
    let c = reshape(c, &[rows, sequence, groups, state_size])?;
    let dt = time_steps(graph, raw_dt.native(), weights.time_bias.native(), shape)?;
    let state = join_state(states, shape, stream)?;
    let [y, next_state] = if sequence == 1 {
        decode(&x, &b, &c, &dt, &state, weights, stream)?
    } else {
        prefill(graph, &x, &b, &c, &dt, &state, weights, shape)?
    };
    let next_state = Array::from_native(graph.astype(&next_state, shape.state_dtype)?)?;
    for (state, value) in states.iter_mut().zip(split_rows(next_state, rows, stream)?) {
        state.value = Some(value);
    }
    let y = graph.reshape(&y, &Shape::new([rows, sequence, heads * head_dim])?)?;
    Array::from_native(y)
}

fn join_state(
    states: &[&mut StateSpaceState],
    shape: MambaShape,
    stream: &Stream,
) -> Result<mirtal::Array> {
    let graph = stream.native().graph();
    let rows = states
        .iter()
        .map(|state| {
            state.value.as_ref().map_or_else(
                || {
                    let dims = Shape::new([1, shape.heads, shape.head_dim, shape.state_size])?;
                    Array::from_native(graph.full(&dims, 0.0, shape.state_dtype)?)
                        .map(StateArray::from)
                },
                StateArray::try_clone,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = rows.iter().collect::<Vec<_>>();
    Ok(StateArray::join(&refs, stream)?.native().clone())
}

/// `dt = clamp(softplus(dt + bias))`, returned as FP32 `[rows, sequence,
/// heads]`.
fn time_steps(
    graph: Graph<'_>,
    raw: &mirtal::Array,
    bias: &mirtal::Array,
    shape: MambaShape,
) -> Result<mirtal::Array> {
    let biased =
        graph.add(&graph.astype(raw, DType::Float32)?, &graph.astype(bias, DType::Float32)?)?;
    let zeros = graph.multiply_scalar(&biased, 0.0)?;
    let softplus = graph.logsumexp(&graph.stack(&[&biased, &zeros], -1)?, -1, false)?;
    let minimum = graph.full(&Shape::new([])?, shape.minimum_time_step, DType::Float32)?;
    Ok(match shape.maximum_time_step {
        Some(value) => {
            let maximum = graph.full(&Shape::new([])?, value, DType::Float32)?;
            graph.clip(&softplus, &minimum, &maximum)?
        },
        None => graph.maximum(&softplus, &minimum)?,
    })
}

fn decode(
    x: &mirtal::Array,
    b: &mirtal::Array,
    c: &mirtal::Array,
    dt: &mirtal::Array,
    state: &mirtal::Array,
    weights: &RecurrenceWeights,
    stream: &Stream,
) -> Result<[mirtal::Array; 2]> {
    let graph = stream.native().graph();
    let skip = graph.astype(weights.skip_scale.native(), x.dtype()?)?;
    stream.state_space_decode([x, weights.decay_log.native(), b, c, &skip, dt, state])
}

#[allow(clippy::too_many_arguments)]
fn prefill(
    graph: Graph<'_>,
    x: &mirtal::Array,
    b: &mirtal::Array,
    c: &mirtal::Array,
    dt: &mirtal::Array,
    state: &mirtal::Array,
    weights: &RecurrenceWeights,
    shape: MambaShape,
) -> Result<[mirtal::Array; 2]> {
    let MambaShape {
        heads,
        head_dim,
        groups,
        state_size,
        chunk,
        ..
    } = shape;
    let dims = x.shape()?;
    let dims = dims.dimensions();
    let (rows, sequence) = (dims[0], dims[1]);
    let decay =
        graph.astype(&graph.negative(&graph.exp(weights.decay_log.native())?)?, DType::Float32)?;
    let dta = graph.multiply(dt, &graph.reshape(&decay, &Shape::new([1, 1, heads])?)?)?;
    let dtx = graph.multiply(&graph.reshape(dt, &Shape::new([rows, sequence, heads, 1])?)?, x)?;
    let (mut outputs, mut carried) = (Vec::with_capacity(sequence.div_ceil(chunk)), state.clone());
    for start in (0..sequence).step_by(chunk) {
        let stop = (start + chunk).min(sequence);
        let window = |array: &mirtal::Array, rank: usize, trailing: &[usize]| {
            let mut from = vec![0; rank];
            from[1] = start;
            let mut to = Vec::with_capacity(rank);
            to.push(rows);
            to.push(stop);
            to.extend_from_slice(trailing);
            graph.slice(array, &from, &to)
        };
        let frame = Frame {
            dtx: window(&dtx, 4, &[heads, head_dim])?,
            dta: window(&dta, 3, &[heads])?,
            b: window(b, 4, &[groups, state_size])?,
            c: window(c, 4, &[groups, state_size])?,
        };
        let (y, next) = chunk_step(graph, &frame, &carried, shape, rows, stop - start)?;
        outputs.push(y);
        carried = next;
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    let y = graph.concatenate(&refs, 1)?;
    let y = graph.astype(&y, x.dtype()?)?;
    let skip = graph.astype(weights.skip_scale.native(), x.dtype()?)?;
    let skip = graph.reshape(&skip, &Shape::new([1, 1, heads, 1])?)?;
    Ok([graph.add(&y, &graph.multiply(x, &skip)?)?, carried])
}

struct Frame {
    dtx: mirtal::Array,
    dta: mirtal::Array,
    b: mirtal::Array,
    c: mirtal::Array,
}

fn chunk_step(
    graph: Graph<'_>,
    frame: &Frame,
    state: &mirtal::Array,
    shape: MambaShape,
    rows: usize,
    length: usize,
) -> Result<(mirtal::Array, mirtal::Array)> {
    let MambaShape { heads, head_dim, groups, state_size, .. } = shape;
    let (dtx, dta, b, c) = (&frame.dtx, &frame.dta, &frame.b, &frame.c);
    let repeats = i32::try_from(heads / groups)?;
    let span = i32::try_from(length)?;
    // Unbatched chunk geometry: [rows, heads, length, length] surrogate kernel.
    let b_col = graph.transpose(b, &[0, 2, 3, 1])?; // [rows, groups, state, length]
    let cb = graph.matmul(&graph.transpose(c, &[0, 2, 1, 3])?, &b_col)?;
    let cb = graph.repeat(&cb, repeats, 1)?;
    let dta_h = graph.transpose(dta, &[0, 2, 1])?; // [rows, heads, length]
    let (lower_strict, lower) = masks(graph, length)?;
    let expanded = graph.repeat(&graph.expand_dims(&dta_h, &[-1])?, span, -1)?;
    let segment =
        graph.cumulative_sum(&graph.multiply(&expanded, &lower_strict)?, -2, false, true)?;
    let decay = graph.exp(&segment)?;
    let surrogate = graph.multiply(&graph.multiply(&cb, &decay)?, &lower)?;
    let y = graph.matmul(&surrogate, &graph.transpose(dtx, &[0, 2, 1, 3])?)?;
    let y = graph.transpose(&y, &[0, 2, 1, 3])?;

    let last = graph.slice(&decay, &[0, 0, length - 1, 0], &[rows, heads, length, length])?;
    let decay_t = graph.transpose(&last, &[0, 3, 1, 2])?;
    let b_step = graph.transpose(&graph.repeat(&b_col, repeats, 1)?, &[0, 1, 3, 2])?;
    let weighted = graph.transpose(&graph.multiply(dtx, &decay_t)?, &[0, 2, 3, 1])?;
    let mut next = graph.matmul(&weighted, &b_step)?;

    let accumulated = graph.exp(&graph.cumulative_sum(dta, -2, false, true)?)?;
    let final_decay = graph.slice(&accumulated, &[0, length - 1, 0], &[rows, length, heads])?;
    let final_decay = graph.reshape(&final_decay, &Shape::new([rows, heads, 1, 1])?)?;
    next = graph.add(&next, &graph.multiply(&final_decay, state)?)?;
    let c_row = graph.reshape(c, &Shape::new([rows, length, groups, 1, state_size, 1])?)?;
    let state_row = graph
        .reshape(state, &Shape::new([rows, 1, groups, heads / groups, head_dim, state_size])?)?;
    let prior = graph.squeeze_axis(&graph.matmul(&state_row, &c_row)?, -1)?;
    let prior = graph.reshape(&prior, &Shape::new([rows, length, heads, head_dim])?)?;
    let y = graph.add(&y, &graph.multiply(&graph.expand_dims(&accumulated, &[-1])?, &prior)?)?;
    Ok((y, next))
}

/// Strict and inclusive lower-triangular masks of `[length, length]`.
fn masks(graph: Graph<'_>, length: usize) -> Result<(mirtal::Array, mirtal::Array)> {
    let stop = f32::from(u16::try_from(length)?);
    let indices = graph.arange(0.0, stop, 1.0, DType::Float32)?;
    let rows = graph.reshape(&indices, &Shape::new([length, 1])?)?;
    let columns = graph.reshape(&indices, &Shape::new([1, length])?)?;
    Ok((graph.less(&columns, &rows)?, graph.greater_equal(&rows, &columns)?))
}
