use super::{
    super::{AttentionWeights, DenseWeights, RouterWeights},
    rope_frequencies,
};
use crate::engine::{
    ModelTensors, NormWeight, QuantizedLinear, Result, Stream, binding::BoundLinear,
    hybrid::moe::HybridMoeLayerConfig,
};

pub(super) fn legacy_norm(tensors: &ModelTensors, prefix: &str, name: &str) -> Result<NormWeight> {
    NormWeight::load(tensors, &format!("{prefix}.{name}"))
}

pub(super) fn legacy_attention(
    tensors: &ModelTensors,
    prefix: &str,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<AttentionWeights> {
    Ok(AttentionWeights {
        query: legacy_linear(tensors, prefix, "q_proj", config.group_size)?,
        key: legacy_linear(tensors, prefix, "k_proj", config.group_size)?,
        value: (!config.use_k_eq_v)
            .then(|| legacy_linear(tensors, prefix, "v_proj", config.group_size))
            .transpose()?,
        output: legacy_linear(tensors, prefix, "o_proj", config.group_size)?,
        query_norm: legacy_norm(tensors, prefix, "q_norm")?,
        key_norm: legacy_norm(tensors, prefix, "k_norm")?,
        rope_frequencies: config
            .proportional_rope
            .then(|| rope_frequencies(config, stream))
            .transpose()?,
    })
}

pub(super) fn legacy_dense(
    tensors: &ModelTensors,
    prefix: &str,
    group: i32,
) -> Result<DenseWeights> {
    Ok(DenseWeights {
        gate: legacy_linear(tensors, prefix, "gate_proj", group)?,
        up: legacy_linear(tensors, prefix, "up_proj", group)?,
        down: legacy_linear(tensors, prefix, "down_proj", group)?,
    })
}

pub(super) fn legacy_router(
    tensors: &ModelTensors,
    prefix: &str,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<RouterWeights> {
    Ok(RouterWeights {
        projection: legacy_linear(tensors, prefix, "proj", config.group_size)?,
        norm_scale: tensors
            .get(&format!("{prefix}.scale"))?
            .multiply_scalar(config.router_norm_scale, stream)?,
        expert_scale: tensors.get(&format!("{prefix}.per_expert_scale"))?,
    })
}

fn legacy_linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group: i32,
) -> Result<BoundLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group).map(BoundLinear::Affine)
}
