use models::weights::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
    Float8ScaleGranularity, Float8ScaleMode,
};

use super::{AdmissionCheckKind, AdmissionStatus};

pub(super) fn metal(format: Float8Quantization) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let activation = match (format.format, format.activation_scale) {
        (Float8Format::E4M3, Float8ActivationScale::None | Float8ActivationScale::DynamicToken)
        | (Float8Format::E5M2, Float8ActivationScale::None) => format.input_scale_dtype.is_none(),
        (Float8Format::E4M3, Float8ActivationScale::StaticTensor) => {
            matches!(
                format.input_scale_dtype,
                Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
            ) && format.input_scale_dtype == format.scale_dtype
        },
        _ => false,
    };
    let identity_scale = format.format == Float8Format::E5M2
        && format.scale_mode == Float8ScaleMode::None
        && format.scale_granularity == Float8ScaleGranularity::None
        && format.scale_dtype.is_none();
    let explicit_scale = matches!(
        format.scale_mode,
        Float8ScaleMode::Multiplier | Float8ScaleMode::InverseMultiplier
    ) && matches!(
        format.scale_granularity,
        Float8ScaleGranularity::Tensor
            | Float8ScaleGranularity::OutputChannel
            | Float8ScaleGranularity::BlockGrid { .. }
    ) && matches!(
        format.scale_dtype,
        Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
    );
    let admitted = activation && (identity_scale || explicit_scale);
    (
        AdmissionCheckKind::Float8,
        if admitted {
            AdmissionStatus::Partial
        } else {
            AdmissionStatus::Unsupported
        },
        "Metal direct FP8 supports E4M3 with dynamic-token or matching-dtype static-tensor activations and E5M2 with BF16 activations; unscaled E5M2 and BF16/F32 tensor, output-channel, or validated block-grid scales are admitted".into(),
    )
}

pub(super) fn cuda(format: Float8Quantization) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let value_format = match format.format {
        Float8Format::E4M3 => matches!(
            format.activation_scale,
            Float8ActivationScale::None
                | Float8ActivationScale::DynamicToken
                | Float8ActivationScale::StaticTensor
        ),
        Float8Format::E5M2 => format.activation_scale == Float8ActivationScale::None,
    };
    let identity_scale = format.format == Float8Format::E5M2
        && format.scale_mode == Float8ScaleMode::None
        && format.scale_granularity == Float8ScaleGranularity::None
        && format.scale_dtype.is_none();
    let explicit_scale = matches!(
        format.scale_mode,
        Float8ScaleMode::Multiplier | Float8ScaleMode::InverseMultiplier
    ) && matches!(
        format.scale_granularity,
        Float8ScaleGranularity::Tensor
            | Float8ScaleGranularity::OutputChannel
            | Float8ScaleGranularity::BlockGrid { .. }
    ) && matches!(
        format.scale_dtype,
        Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
    );
    let activation_scale = format.activation_scale != Float8ActivationScale::StaticTensor
        || (matches!(
            format.input_scale_dtype,
            Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
        ) && format.input_scale_dtype == format.scale_dtype);
    let admitted = value_format && activation_scale && (identity_scale || explicit_scale);
    (
        AdmissionCheckKind::Float8,
        if admitted {
            AdmissionStatus::Partial
        } else {
            AdmissionStatus::Unsupported
        },
        "CUDA direct FP8 supports unscaled E4M3/E5M2 weights or BF16/F32 tensor, output-channel, and exact-divisible block-grid scales; dynamic-token and matching-dtype static activation scaling are E4M3-only".into(),
    )
}
