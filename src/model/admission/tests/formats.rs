use foundation::model::BackendTarget;
use models::weights::{
    AffineBits, BlockQuantization, Float8ActivationScale, Float8Format, Float8ParameterDType,
    Float8Quantization, Float8ScaleGranularity, Float8ScaleMode,
};

use super::{affine, packed_int4, packed_int8};
use crate::model::admission::{AdmissionStatus, WeightEncoding, registry};

#[test]
fn current_format_matrix_is_explicit() {
    let cases = [
        (
            BackendTarget::Metal,
            WeightEncoding::Dense { dtype: "BF16".into() },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::Affine { format: affine(AffineBits::Four) },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::Affine { format: affine(AffineBits::Three) },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Metal,
            WeightEncoding::PackedInt8 { format: packed_int8() },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::PackedInt8 { format: packed_int8() },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Metal,
            WeightEncoding::PackedInt4 { format: packed_int4() },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::PackedInt4 { format: packed_int4() },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::Float8 {
                format: Float8Quantization::unscaled(Float8Format::E4M3),
            },
            AdmissionStatus::Unsupported,
        ),
        (
            BackendTarget::Metal,
            WeightEncoding::MxFp4 { format: BlockQuantization::MXFP4 },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Metal,
            WeightEncoding::MxFp8 { format: BlockQuantization::MXFP8 },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::MxFp8 { format: BlockQuantization::MXFP8 },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Cuda,
            WeightEncoding::NvFp4 { format: BlockQuantization::NVFP4 },
            AdmissionStatus::Partial,
        ),
        (
            BackendTarget::Metal,
            WeightEncoding::NvFp4 { format: BlockQuantization::NVFP4 },
            AdmissionStatus::Partial,
        ),
    ];

    for (backend, encoding, expected) in cases {
        assert_eq!(registry::assess(&backend, &encoding).status, expected);
    }
}

#[test]
fn direct_e4m3_output_channel_is_admitted_with_backend_specific_gates() {
    let encoding = WeightEncoding::Float8 {
        format: Float8Quantization {
            format: Float8Format::E4M3,
            scale_mode: Float8ScaleMode::Multiplier,
            scale_granularity: Float8ScaleGranularity::OutputChannel,
            scale_dtype: Some(Float8ParameterDType::F32),
            activation_scale: Float8ActivationScale::DynamicToken,
            input_scale_dtype: None,
        },
    };
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &encoding).status,
        AdmissionStatus::Partial
    );
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &encoding).status,
        AdmissionStatus::Partial
    );
}

#[test]
fn direct_fp8_block_grid_is_admitted_on_both_accelerators() {
    let encoding = WeightEncoding::Float8 {
        format: Float8Quantization {
            format: Float8Format::E4M3,
            scale_mode: Float8ScaleMode::Multiplier,
            scale_granularity: Float8ScaleGranularity::BlockGrid {
                output_groups: 2,
                input_groups: 4,
                output_block_size: None,
                input_block_size: None,
            },
            scale_dtype: Some(Float8ParameterDType::F32),
            activation_scale: Float8ActivationScale::None,
            input_scale_dtype: None,
        },
    };
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &encoding).status,
        AdmissionStatus::Partial
    );
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &encoding).status,
        AdmissionStatus::Partial
    );
}

#[test]
fn direct_e5m2_bf16_activation_is_admitted_on_both_accelerators() {
    let scaled = WeightEncoding::Float8 {
        format: Float8Quantization {
            format: Float8Format::E5M2,
            scale_mode: Float8ScaleMode::Multiplier,
            scale_granularity: Float8ScaleGranularity::Tensor,
            scale_dtype: Some(Float8ParameterDType::F32),
            activation_scale: Float8ActivationScale::None,
            input_scale_dtype: None,
        },
    };
    let unscaled = WeightEncoding::Float8 {
        format: Float8Quantization::unscaled(Float8Format::E5M2),
    };
    for encoding in [scaled, unscaled] {
        assert_eq!(
            registry::assess(&BackendTarget::Cuda, &encoding).status,
            AdmissionStatus::Partial
        );
        assert_eq!(
            registry::assess(&BackendTarget::Metal, &encoding).status,
            AdmissionStatus::Partial
        );
    }
}

#[test]
fn static_e4m3_activation_requires_matching_explicit_scale() {
    let mut format = Float8Quantization {
        format: Float8Format::E4M3,
        scale_mode: Float8ScaleMode::Multiplier,
        scale_granularity: Float8ScaleGranularity::Tensor,
        scale_dtype: Some(Float8ParameterDType::F32),
        activation_scale: Float8ActivationScale::StaticTensor,
        input_scale_dtype: Some(Float8ParameterDType::F32),
    };
    let encoding = WeightEncoding::Float8 { format };
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &encoding).status,
        AdmissionStatus::Partial
    );
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &encoding).status,
        AdmissionStatus::Partial
    );
    format.input_scale_dtype = Some(Float8ParameterDType::BF16);
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &WeightEncoding::Float8 { format }).status,
        AdmissionStatus::Unsupported
    );
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &WeightEncoding::Float8 { format }).status,
        AdmissionStatus::Unsupported
    );
    format.scale_dtype = Some(Float8ParameterDType::BF16);
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &WeightEncoding::Float8 { format }).status,
        AdmissionStatus::Partial
    );
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &WeightEncoding::Float8 { format }).status,
        AdmissionStatus::Partial
    );
    let mut unscaled = Float8Quantization::unscaled(Float8Format::E4M3);
    unscaled.activation_scale = Float8ActivationScale::StaticTensor;
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &WeightEncoding::Float8 { format: unscaled }).status,
        AdmissionStatus::Unsupported
    );
}
