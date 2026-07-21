use models::weights::{
    RoutedDecoderLayerBindings, RoutedExpertBindings, TensorBinding, TensorStorage,
};

use super::{ClampedRoutedConfig, ClampedRoutedExpertWeights, MlxExpertWeights, affine, tensor};
use crate::{
    CudaTensorSet, Result,
    backend::clamped_routed::{
        projection::{ClampedRoutedLinearWeight, ClampedRoutedQkvProjections},
        validation::validate_mlx_experts,
    },
};

pub(super) fn load(
    config: ClampedRoutedConfig,
    tensors: &CudaTensorSet,
    bindings: RoutedDecoderLayerBindings<'_>,
) -> Result<(
    ClampedRoutedQkvProjections,
    ClampedRoutedLinearWeight,
    ClampedRoutedLinearWeight,
    ClampedRoutedExpertWeights,
)> {
    let q = affine(tensors, bindings.query)?;
    let k = affine(tensors, bindings.key)?;
    let v = affine(tensors, bindings.value)?;
    q.infer_config(1, config.hidden, config.query_heads * config.head_dim)?;
    k.infer_config(1, config.hidden, config.kv_heads * config.head_dim)?;
    v.infer_config(1, config.hidden, config.kv_heads * config.head_dim)?;
    let output = affine(tensors, bindings.attention_output)?;
    output.infer_config(1, config.query_heads * config.head_dim, config.hidden)?;
    let router = affine(tensors, bindings.router)?;
    router.infer_config(1, config.hidden, config.experts)?;
    let RoutedExpertBindings::SeparateGateUp { gate, up, down } = bindings.experts else {
        return Err(crate::Error::InvalidDecoderKernel(
            "MLX clamped-routed requires separate gate/up expert bindings",
        ));
    };
    let (gate_scales, gate_bias) = expert_companions(gate)?;
    let (up_scales, up_bias) = expert_companions(up)?;
    let (down_scales, down_bias) = expert_companions(down)?;
    let experts = ClampedRoutedExpertWeights::Mlx(Box::new(MlxExpertWeights {
        gate_blocks: tensor(tensors, &gate.source)?,
        gate_scales: tensor(tensors, gate_scales)?,
        gate_bias: tensor(tensors, gate_bias)?,
        up_blocks: tensor(tensors, &up.source)?,
        up_scales: tensor(tensors, up_scales)?,
        up_bias: tensor(tensors, up_bias)?,
        down_blocks: tensor(tensors, &down.source)?,
        down_scales: tensor(tensors, down_scales)?,
        down_bias: tensor(tensors, down_bias)?,
    }));
    validate_mlx_experts(config, &experts)?;
    Ok((
        ClampedRoutedQkvProjections::Mlx(Box::new([q, k, v])),
        ClampedRoutedLinearWeight::Mlx(output),
        ClampedRoutedLinearWeight::Mlx(router),
        experts,
    ))
}

fn expert_companions(binding: &TensorBinding) -> Result<(&str, &str)> {
    let TensorStorage::AffineQuantized { scales, output_bias: Some(bias), .. } = &binding.storage
    else {
        return Err(crate::Error::InvalidDecoderKernel(
            "MLX clamped-routed expert binding requires scales and output bias",
        ));
    };
    Ok((scales, bias))
}
