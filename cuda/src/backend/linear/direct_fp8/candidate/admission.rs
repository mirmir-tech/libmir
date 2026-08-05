use mircuda::{DeviceBuffer, ScaledFp8Scale, bf16};

use super::{
    CudaBackend, DirectFp8Activation, DirectFp8CheckpointWeight, DirectFp8Format, DirectFp8Scale,
    DirectFp8Spec, Error, Result,
};

pub(super) fn cublaslt_admitted(
    spec: DirectFp8Spec,
    scale: Option<ScaledFp8Scale>,
    has_bias: bool,
) -> bool {
    spec.tokens == 1
        && spec.format == DirectFp8Format::E4M3
        && spec.scale == DirectFp8Scale::Tensor
        && spec.activation == DirectFp8Activation::StaticE4M3Tensor
        && matches!(scale, Some(ScaledFp8Scale::F32))
        && !has_bias
}

pub(super) fn bias(weight: &DirectFp8CheckpointWeight) -> Result<Option<&DeviceBuffer<bf16>>> {
    weight
        .bias
        .as_ref()
        .map(|value| {
            value.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                name: value.name().into(),
                expected: "BF16",
            })
        })
        .transpose()
}

pub(in crate::backend::linear::direct_fp8) fn tensor_core_admitted(
    backend: &CudaBackend,
    spec: DirectFp8Spec,
    scale: Option<ScaledFp8Scale>,
) -> bool {
    let scaled_e4m3 = scale.is_some()
        && spec.format == DirectFp8Format::E4M3
        && matches!(
            (spec.scale, spec.activation),
            (DirectFp8Scale::OutputChannel, DirectFp8Activation::DynamicE4M3Token)
                | (
                    DirectFp8Scale::Tensor | DirectFp8Scale::OutputChannel,
                    DirectFp8Activation::StaticE4M3Tensor
                )
        );
    let weight_only_e5m2 = scale.is_none()
        && spec.format == DirectFp8Format::E5M2
        && spec.scale == DirectFp8Scale::Tensor
        && spec.activation == DirectFp8Activation::Bf16
        && spec.input_features.is_multiple_of(16)
        && spec.output_features.is_multiple_of(16);
    backend.inner.device.compute_capability.0 == 12
        && !spec.inverse_scale
        && (scaled_e4m3 || weight_only_e5m2)
}
