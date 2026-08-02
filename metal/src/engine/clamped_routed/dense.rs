use super::projection::BoundLinear;
use crate::engine::{Array, Result, RouterOutput, Stream, kernels::MxFp4Shape};

pub(super) fn dense_experts(
    input: &Array,
    routing: &RouterOutput,
    projections: [&BoundLinear; 3],
    limit: &Array,
    shape: MxFp4Shape,
    stream: &Stream,
) -> Result<Array> {
    let [gate, up, down] = projections;
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
    let input = expert_input(input, shape, stream)?;
    let gate_up = gate_up.gather(&input, &routing.indices, false, stream)?;
    let (gate, up) = if interleaved {
        super::super::fused_gate_up::split_interleaved_last(&gate_up, shape.intermediate, stream)?
    } else {
        super::super::fused_gate_up::split_last(&gate_up, shape.intermediate, stream)?
    };
    activate_and_down(&gate, &up, down, routing, limit, stream)
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
    let minimum = limit.multiply_scalar(-1.0, stream)?;
    let graph = stream.native().graph();
    let gate = Array::from_native(graph.minimum(gate.native(), limit.native())?)?;
    let up = up.clip(&minimum, limit, stream)?.add_scalar(1.0, stream)?;
    let scaled = gate.multiply_scalar(1.702, stream)?;
    let silu =
        Array::from_native(graph.multiply(gate.native(), &graph.sigmoid(scaled.native())?)?)?;
    let activated = silu.multiply(&up, stream)?;
    down.gather(&activated, &routing.indices, false, stream)?
        .squeeze_axis(-2, stream)?
        .weighted_sum(&routing.weights, -2, stream)
}
