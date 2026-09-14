use super::{Array, BoundLinear, MxFp4Shape, Result, RouterOutput, Stream, activate};

/// Test-only sorted prefill, using the same route permutation and restoration
/// as Qwen while retaining clamped activation, gathered biases and graph sum.
pub(super) fn forward(
    input: &Array,
    routing: &RouterOutput,
    down: &BoundLinear,
    limit: &Array,
    shape: MxFp4Shape,
    stream: &Stream,
    gate_up: impl FnOnce(&Array, &Array) -> Result<(Array, Array)>,
) -> Result<Array> {
    let tokens = i32::try_from(shape.tokens)?;
    let hidden = i32::try_from(shape.hidden)?;
    let top_k = i32::try_from(shape.top_k)?;
    let input = input.reshape(&[1, tokens, hidden], stream)?;
    let indices = routing.indices.reshape(&[1, tokens, top_k], stream)?;
    let weights = routing.weights.reshape(&[1, tokens, top_k], stream)?;
    let sorted = input.sort_expert_inputs(&indices, stream)?;
    let (gate, up) = gate_up(&sorted.input, &sorted.indices)?;
    let activated = activate(&gate, &up, limit, stream)?;
    let output = down.gather(&activated, &sorted.indices, true, stream)?;
    sorted
        .restore(&output, stream)?
        .weighted_sum(&weights, -2, stream)?
        .reshape(&[tokens, hidden], stream)
}
