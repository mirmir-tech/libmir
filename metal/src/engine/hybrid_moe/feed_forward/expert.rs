use super::{HybridMoeLayerConfig, LayerWeights, routing};
use crate::engine::{
    Array, FusedExpertGateUp, Result, Stream, expert_tuning,
    route_tuning::{self, ExpertActivation, RoutingSpec},
};

pub(in crate::engine::hybrid_moe) fn experts(
    input: &Array,
    weights: &LayerWeights,
    config: HybridMoeLayerConfig,
    fused_gate_up: Option<&FusedExpertGateUp>,
    stream: &Stream,
) -> Result<Array> {
    let routing = routing(input, weights, config, stream)?;
    let hidden = weights.pre_expert_norm.apply(input, config.rms_norm_eps, stream)?;
    let (group_size, bits) = weights.experts.tuning_format();
    let spec = RoutingSpec {
        experts: config.expert_count,
        intermediate: config.expert_intermediate,
        group_size,
        bits,
        activation: ExpertActivation::GeluApprox,
        fused_unsorted: fused_gate_up.is_some(),
    };
    let output = route_tuning::forward(
        spec,
        &hidden,
        &routing.indices,
        stream,
        (
            |indices| {
                let sorted = hidden.sort_expert_inputs(indices, stream)?;
                let output = expert_mlp(&sorted.input, &sorted.indices, weights, true, stream)?;
                sorted.restore(&output, stream)?.weighted_sum(&routing.weights, -2, stream)
            },
            |indices| {
                let sorted = hidden.sort_expert_inputs(indices, stream)?;
                let output = expert_mlp(&sorted.input, &sorted.indices, weights, true, stream)?;
                sorted.restore_weighted(&output, &routing.weights, stream)
            },
            |indices| {
                let grouped = hidden.group_expert_inputs(indices, config.expert_count, stream)?;
                let output = expert_mlp(&grouped.input, &grouped.indices, weights, true, stream)?;
                grouped.restore_weighted(&output, &routing.weights, stream)
            },
            |indices| {
                unsorted(&hidden, indices, weights, fused_gate_up, false, stream)?
                    .weighted_sum(&routing.weights, -2, stream)
            },
            |indices| {
                unsorted(&hidden, indices, weights, fused_gate_up, true, stream)?
                    .weighted_sum(&routing.weights, -2, stream)
            },
        ),
    )?;
    weights.post_expert_norm.apply(&output, config.rms_norm_eps, stream)
}

fn unsorted(
    hidden: &Array,
    indices: &Array,
    weights: &LayerWeights,
    fused_gate_up: Option<&FusedExpertGateUp>,
    native_output: bool,
    stream: &Stream,
) -> Result<Array> {
    let hidden = hidden.expand_dims(&[-2, -3], stream)?;
    let (gate, up) = match (fused_gate_up, weights.experts.separate()) {
        (Some(fused), Some((gate, up))) => {
            expert_tuning::forward(gate, up, fused, &hidden, indices, stream)?
        },
        _ if native_output => weights.experts.gather_gate_up_native(&hidden, indices, stream)?,
        _ => weights.experts.gather_gate_up(&hidden, indices, false, stream)?,
    };
    let activated = gate.gelu_approx_mul(&up, stream)?;
    let output = if native_output {
        weights.experts.down.gather_native(&activated, indices, false, stream)?
    } else {
        weights.experts.down.gather(&activated, indices, false, stream)?
    };
    output.squeeze_axis(-2, stream)
}

fn expert_mlp(
    input: &Array,
    indices: &Array,
    weights: &LayerWeights,
    sorted_indices: bool,
    stream: &Stream,
) -> Result<Array> {
    let (gate, up) = weights.experts.gather_gate_up(input, indices, sorted_indices, stream)?;
    let activated = gate.gelu_approx_mul(&up, stream)?;
    weights.experts.down.gather(&activated, indices, sorted_indices, stream)
}
