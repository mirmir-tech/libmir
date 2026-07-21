use super::{ClampedRoutedConfig, weights::ClampedRoutedExpertWeights};
use crate::{CudaTensor, CudaTensorDType, Error, Result};

pub(super) fn validate_common(
    config: ClampedRoutedConfig,
    tensors: [&CudaTensor; 8],
) -> Result<()> {
    let q = config.query_heads * config.head_dim;
    let kv = config.kv_heads * config.head_dim;
    let shapes: [&[usize]; 8] = [
        &[config.hidden],
        &[q],
        &[kv],
        &[kv],
        &[config.hidden],
        &[config.query_heads],
        &[config.hidden],
        &[config.experts],
    ];
    for (tensor, shape) in tensors.into_iter().zip(shapes) {
        validate(tensor, shape, CudaTensorDType::Bf16)?;
    }
    Ok(())
}

pub(super) fn validate_native_experts(
    config: ClampedRoutedConfig,
    weights: &ClampedRoutedExpertWeights,
) -> Result<()> {
    let ClampedRoutedExpertWeights::Native(weights) = weights else {
        return Err(Error::InvalidExecutionPlan("expected native clamped-routed experts"));
    };
    let gate_rows = 2 * config.intermediate;
    validate(
        &weights.gate_up_blocks,
        &[config.experts, gate_rows, config.hidden / 32, 16],
        CudaTensorDType::U8,
    )?;
    validate(
        &weights.gate_up_scales,
        &[config.experts, gate_rows, config.hidden / 32],
        CudaTensorDType::U8,
    )?;
    validate(&weights.gate_up_bias, &[config.experts, gate_rows], CudaTensorDType::Bf16)?;
    validate(
        &weights.down_blocks,
        &[config.experts, config.hidden, config.intermediate / 32, 16],
        CudaTensorDType::U8,
    )?;
    validate(
        &weights.down_scales,
        &[config.experts, config.hidden, config.intermediate / 32],
        CudaTensorDType::U8,
    )?;
    validate(&weights.down_bias, &[config.experts, config.hidden], CudaTensorDType::Bf16)
}

pub(super) fn validate_mlx_experts(
    config: ClampedRoutedConfig,
    weights: &ClampedRoutedExpertWeights,
) -> Result<()> {
    let ClampedRoutedExpertWeights::Mlx(weights) = weights else {
        return Err(Error::InvalidExecutionPlan("expected MLX clamped-routed experts"));
    };
    let gate_shape = [config.experts, config.intermediate, config.hidden / 8];
    let gate_scale_shape = [config.experts, config.intermediate, config.hidden / 32];
    let gate_bias_shape = [config.experts, config.intermediate];
    for tensor in [&weights.gate_blocks, &weights.up_blocks] {
        validate(tensor, &gate_shape, CudaTensorDType::U32)?;
    }
    for tensor in [&weights.gate_scales, &weights.up_scales] {
        validate(tensor, &gate_scale_shape, CudaTensorDType::U8)?;
    }
    for tensor in [&weights.gate_bias, &weights.up_bias] {
        validate(tensor, &gate_bias_shape, CudaTensorDType::Bf16)?;
    }
    validate(
        &weights.down_blocks,
        &[config.experts, config.hidden, config.intermediate / 8],
        CudaTensorDType::U32,
    )?;
    validate(
        &weights.down_scales,
        &[config.experts, config.hidden, config.intermediate / 32],
        CudaTensorDType::U8,
    )?;
    validate(&weights.down_bias, &[config.experts, config.hidden], CudaTensorDType::Bf16)
}

fn validate(tensor: &CudaTensor, shape: &[usize], dtype: CudaTensorDType) -> Result<()> {
    if tensor.shape() != shape {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    if tensor.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: match dtype {
                CudaTensorDType::Bf16 => "BF16",
                CudaTensorDType::U32 => "U32",
                _ => "U8",
            },
        });
    }
    Ok(())
}
