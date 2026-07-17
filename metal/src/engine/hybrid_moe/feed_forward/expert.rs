use super::{HybridMoeLayerConfig, LayerWeights, routing};
use crate::engine::{Array, Error, FusedExpertGateUp, Result, Stream};

pub(in crate::engine::hybrid_moe) fn experts(
    input: &Array,
    weights: &LayerWeights,
    config: HybridMoeLayerConfig,
    fused_gate_up: Option<&FusedExpertGateUp>,
    stream: &Stream,
) -> Result<Array> {
    let routing = routing(input, weights, config, stream)?;
    let hidden = weights.pre_expert_norm.apply(input, config.rms_norm_eps, stream)?;
    let output = if sort_experts(&routing.indices)? {
        let sorted = hidden.sort_expert_inputs(&routing.indices, stream)?;
        let output = expert_mlp(&sorted.input, &sorted.indices, weights, true, stream)?;
        sorted.restore(&output, stream)?
    } else {
        let hidden = hidden.expand_dims(&[-2, -3], stream)?;
        let (gate, up) = fused_gate_up.map_or_else(
            || {
                Ok((
                    weights.experts.gate.gather(&hidden, &routing.indices, false, stream)?,
                    weights.experts.up.gather(&hidden, &routing.indices, false, stream)?,
                ))
            },
            |fused| fused.forward(&hidden, &routing.indices, stream),
        )?;
        let activated = gate.gelu_approx_mul(&up, stream)?;
        weights
            .experts
            .down
            .gather(&activated, &routing.indices, false, stream)?
            .squeeze_axis(-2, stream)?
    };
    let output = output.weighted_sum(&routing.weights, -2, stream)?;
    weights.post_expert_norm.apply(&output, config.rms_norm_eps, stream)
}

fn expert_mlp(
    input: &Array,
    indices: &Array,
    weights: &LayerWeights,
    sorted_indices: bool,
    stream: &Stream,
) -> Result<Array> {
    let up = weights.experts.up.gather(input, indices, sorted_indices, stream)?;
    let gate = weights.experts.gate.gather(input, indices, sorted_indices, stream)?;
    let activated = gate.gelu_approx_mul(&up, stream)?;
    weights.experts.down.gather(&activated, indices, sorted_indices, stream)
}

fn sort_experts(indices: &Array) -> Result<bool> {
    let mut count = 1_usize;
    for dimension in indices.shape()? {
        count = count.checked_mul(usize::try_from(dimension)?).ok_or(Error::ShapeOverflow)?;
    }
    Ok(count >= 64)
}
