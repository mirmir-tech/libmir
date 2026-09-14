use super::projection::BoundLinear;
use crate::engine::{Array, Result, RouterOutput, Stream, kernels::MxFp4Shape};

#[cfg(test)]
mod prefill;
#[cfg(test)]
mod tests;

pub(super) fn dense_experts(
    input: &Array,
    routing: &RouterOutput,
    projections: [&BoundLinear; 3],
    limit: &Array,
    shape: MxFp4Shape,
    stream: &Stream,
) -> Result<Array> {
    let [gate, up, down] = projections;
    #[cfg(test)]
    if shape.tokens > 1
        && stream.config().diagnostics.moe_prefill == crate::config::MoePrefill::ClampedSorted
    {
        return prefill::forward(input, routing, down, limit, shape, stream, |input, indices| {
            Ok((
                gate.gather(input, indices, true, stream)?,
                up.gather(input, indices, true, stream)?,
            ))
        });
    }
    let input = expert_input(input, shape, stream)?;
    let gate = gate.gather(&input, &routing.indices, false, stream)?;
    let up = up.gather(&input, &routing.indices, false, stream)?;
    activate_and_down(&gate, &up, down, routing, limit, stream)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_dense_experts(
    input: &Array,
    routing: &RouterOutput,
    gate_up: &BoundLinear,
    down: &BoundLinear,
    interleaved: bool,
    limit: &Array,
    shape: MxFp4Shape,
    stream: &Stream,
) -> Result<Array> {
    #[cfg(test)]
    if shape.tokens > 1
        && stream.config().diagnostics.moe_prefill == crate::config::MoePrefill::ClampedSorted
    {
        return prefill::forward(input, routing, down, limit, shape, stream, |input, indices| {
            let output = gate_up.gather(input, indices, true, stream)?;
            split(&output, interleaved, shape.intermediate, stream)
        });
    }
    let input = expert_input(input, shape, stream)?;
    let gate_up = gate_up.gather(&input, &routing.indices, false, stream)?;
    let (gate, up) = split(&gate_up, interleaved, shape.intermediate, stream)?;
    activate_and_down(&gate, &up, down, routing, limit, stream)
}

fn split(
    output: &Array,
    interleaved: bool,
    intermediate: usize,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if interleaved {
        super::super::fused_gate_up::split_interleaved_last(output, intermediate, stream)
    } else {
        super::super::fused_gate_up::split_last(output, intermediate, stream)
    }
}

fn expert_input(input: &Array, shape: MxFp4Shape, stream: &Stream) -> Result<Array> {
    input.reshape(&[i32::try_from(shape.tokens)?, 1, 1, i32::try_from(shape.hidden)?], stream)
}

fn activate_and_down(
    gate: &Array,
    up: &Array,
    down: &BoundLinear,
    routing: &RouterOutput,
    limit: &Array,
    stream: &Stream,
) -> Result<Array> {
    let activated = activate(gate, up, limit, stream)?;
    down.gather(&activated, &routing.indices, false, stream)?
        .squeeze_axis(-2, stream)?
        .weighted_sum(&routing.weights, -2, stream)
}

fn activate(gate: &Array, up: &Array, limit: &Array, stream: &Stream) -> Result<Array> {
    let limit = limit.astype_like(gate, stream)?;
    let minimum = limit.multiply_scalar(-1.0, stream)?;
    let graph = stream.native().graph();
    let gate = Array::from_native(graph.minimum(gate.native(), limit.native())?)?;
    let up = up.clip(&minimum, &limit, stream)?.add_scalar(1.0, stream)?;
    let scaled = gate.multiply_scalar(1.702, stream)?;
    let silu =
        Array::from_native(graph.multiply(gate.native(), &graph.sigmoid(scaled.native())?)?)?;
    silu.multiply(&up, stream)
}
