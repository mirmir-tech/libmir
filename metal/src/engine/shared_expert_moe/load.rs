use crate::engine::{
    ModelTensors, QuantizedLinear, Result, binding::BoundLinear, fusion_planner::ProjectionBiases,
};

pub(super) fn projection_biases(gate: &BoundLinear, up: &BoundLinear) -> ProjectionBiases {
    ProjectionBiases::new([false, false], None, [gate.has_bias(), up.has_bias()])
}

pub(super) fn linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group_size: i32,
) -> Result<BoundLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size).map(BoundLinear::Affine)
}
