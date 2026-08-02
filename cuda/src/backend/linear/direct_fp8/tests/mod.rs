use models::weights::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
    Float8ScaleGranularity, Float8ScaleMode,
};

use super::{DirectFp8Activation, DirectFp8Format, DirectFp8Scale, Result, execution_contract};

#[cfg(target_os = "linux")]
mod autotune;

#[test]
fn maps_only_unambiguous_direct_fp8_scale_contracts() -> Result<()> {
    let tensor = contract(Float8ScaleGranularity::Tensor, Float8ScaleMode::Multiplier);
    assert_eq!(
        execution_contract("weight", tensor, 128, 64)?,
        (DirectFp8Format::E4M3, DirectFp8Scale::Tensor, false, DirectFp8Activation::Bf16,)
    );
    let inverse =
        contract(Float8ScaleGranularity::OutputChannel, Float8ScaleMode::InverseMultiplier);
    assert_eq!(
        execution_contract("weight", inverse, 128, 64)?,
        (
            DirectFp8Format::E4M3,
            DirectFp8Scale::OutputChannel,
            true,
            DirectFp8Activation::Bf16,
        )
    );
    let mut bf16 = tensor;
    bf16.scale_dtype = Some(Float8ParameterDType::BF16);
    assert_eq!(
        execution_contract("weight", bf16, 128, 64)?,
        (DirectFp8Format::E4M3, DirectFp8Scale::Tensor, false, DirectFp8Activation::Bf16,)
    );
    let mut e5m2 = tensor;
    e5m2.format = Float8Format::E5M2;
    assert_eq!(
        execution_contract("weight", e5m2, 128, 64)?,
        (DirectFp8Format::E5M2, DirectFp8Scale::Tensor, false, DirectFp8Activation::Bf16,)
    );
    e5m2.activation_scale = Float8ActivationScale::DynamicToken;
    assert!(execution_contract("weight", e5m2, 128, 64).is_err());
    let mut static_e4m3 = tensor;
    static_e4m3.activation_scale = Float8ActivationScale::StaticTensor;
    static_e4m3.input_scale_dtype = Some(Float8ParameterDType::F32);
    assert_eq!(
        execution_contract("weight", static_e4m3, 128, 64)?,
        (
            DirectFp8Format::E4M3,
            DirectFp8Scale::Tensor,
            false,
            DirectFp8Activation::StaticE4M3Tensor,
        )
    );
    static_e4m3.input_scale_dtype = Some(Float8ParameterDType::BF16);
    assert!(execution_contract("weight", static_e4m3, 128, 64).is_err());
    static_e4m3.scale_dtype = Some(Float8ParameterDType::BF16);
    assert_eq!(
        execution_contract("weight", static_e4m3, 128, 64)?.3,
        DirectFp8Activation::StaticE4M3Tensor
    );
    let unscaled = Float8Quantization::unscaled(Float8Format::E5M2);
    assert_eq!(
        execution_contract("weight", unscaled, 128, 64)?,
        (DirectFp8Format::E5M2, DirectFp8Scale::Tensor, false, DirectFp8Activation::Bf16,)
    );
    let mut incomplete = unscaled;
    incomplete.scale_dtype = Some(Float8ParameterDType::F32);
    assert!(execution_contract("weight", incomplete, 128, 64).is_err());
    let mut unscaled_static = Float8Quantization::unscaled(Float8Format::E4M3);
    unscaled_static.activation_scale = Float8ActivationScale::StaticTensor;
    assert!(execution_contract("weight", unscaled_static, 128, 64).is_err());
    Ok(())
}

#[test]
fn maps_exact_and_declared_padded_block_grids() -> Result<()> {
    let block = contract(
        Float8ScaleGranularity::BlockGrid {
            output_groups: 64,
            input_groups: 4,
            output_block_size: None,
            input_block_size: None,
        },
        Float8ScaleMode::Multiplier,
    );
    assert_eq!(
        execution_contract("weight", block, 128, 64)?,
        (
            DirectFp8Format::E4M3,
            DirectFp8Scale::BlockGrid {
                output_groups: 64,
                input_groups: 4,
                output_block_size: 1,
                input_block_size: 32,
            },
            false,
            DirectFp8Activation::Bf16,
        )
    );
    let exact_grid = contract(
        Float8ScaleGranularity::BlockGrid {
            output_groups: 2,
            input_groups: 4,
            output_block_size: None,
            input_block_size: None,
        },
        Float8ScaleMode::Multiplier,
    );
    assert_eq!(
        execution_contract("weight", exact_grid, 128, 64)?,
        (
            DirectFp8Format::E4M3,
            DirectFp8Scale::BlockGrid {
                output_groups: 2,
                input_groups: 4,
                output_block_size: 32,
                input_block_size: 32,
            },
            false,
            DirectFp8Activation::Bf16,
        )
    );
    let ambiguous = contract(
        Float8ScaleGranularity::BlockGrid {
            output_groups: 3,
            input_groups: 4,
            output_block_size: None,
            input_block_size: None,
        },
        Float8ScaleMode::Multiplier,
    );
    assert!(execution_contract("weight", ambiguous, 128, 64).is_err());
    let padded = contract(
        Float8ScaleGranularity::BlockGrid {
            output_groups: 3,
            input_groups: 2,
            output_block_size: Some(32),
            input_block_size: Some(128),
        },
        Float8ScaleMode::Multiplier,
    );
    assert_eq!(
        execution_contract("weight", padded, 132, 65)?.1,
        DirectFp8Scale::BlockGrid {
            output_groups: 3,
            input_groups: 2,
            output_block_size: 32,
            input_block_size: 128,
        }
    );
    Ok(())
}

fn contract(
    scale_granularity: Float8ScaleGranularity,
    scale_mode: Float8ScaleMode,
) -> Float8Quantization {
    Float8Quantization {
        format: Float8Format::E4M3,
        scale_mode,
        scale_granularity,
        scale_dtype: Some(Float8ParameterDType::F32),
        activation_scale: Float8ActivationScale::None,
        input_scale_dtype: None,
    }
}
