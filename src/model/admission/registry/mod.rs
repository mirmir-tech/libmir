use foundation::model::BackendTarget;
use models::weights::{
    AffineParameterDType, AffineStorageDType, CompressedIntegerQuantization,
    CompressedIntegerScaleDType, GptqBits, GptqQuantization, GptqScaleDType,
    GroupedAffineQuantization,
};

use super::{AdmissionCheck, AdmissionCheckKind, AdmissionStatus, WeightEncoding};

mod bitsandbytes;
mod float8;
mod kind;

use bitsandbytes::assess as bitsandbytes;
use float8::{cuda as float8_cuda, metal as float8_metal};
use kind::kind;

pub(super) fn assess(backend: &BackendTarget, encoding: &WeightEncoding) -> AdmissionCheck {
    let (kind, status, detail) = match (backend, encoding) {
        (BackendTarget::CpuReference, _) => (
            kind(encoding),
            AdmissionStatus::Unsupported,
            "the CPU reference backend does not execute product models".into(),
        ),
        (BackendTarget::Metal, WeightEncoding::Dense { dtype }) => {
            dense(dtype, "Metal execution depends on the admitted architecture")
        },
        (BackendTarget::Cuda, WeightEncoding::Dense { dtype }) => {
            dense(dtype, "CUDA execution depends on the admitted architecture")
        },
        (BackendTarget::Metal, WeightEncoding::Affine { format }) => affine_metal(*format),
        (BackendTarget::Cuda, WeightEncoding::Affine { format }) => affine_cuda(*format),
        (BackendTarget::Metal, WeightEncoding::PackedInt8 { format }) => {
            packed_int8("Metal", *format)
        },
        (BackendTarget::Cuda, WeightEncoding::PackedInt8 { format }) => {
            packed_int8("CUDA", *format)
        },
        (BackendTarget::Metal, WeightEncoding::PackedInt4 { format }) => {
            packed_int4("Metal", *format)
        },
        (BackendTarget::Cuda, WeightEncoding::PackedInt4 { format }) => {
            packed_int4("CUDA", *format)
        },
        (BackendTarget::Metal, WeightEncoding::Awq { format }) => (
            AdmissionCheckKind::Awq,
            if format.is_gemm_w4a16() {
                AdmissionStatus::Partial
            } else {
                AdmissionStatus::Unsupported
            },
            format!(
                "Metal requires AWQ GEMM W4A16 for device repack to native affine: G{} {:?}",
                format.group_size, format.packing
            ),
        ),
        (BackendTarget::Cuda, WeightEncoding::Awq { format }) => (
            AdmissionCheckKind::Awq,
            if format.is_gemm_w4a16() {
                AdmissionStatus::Partial
            } else {
                AdmissionStatus::Unsupported
            },
            format!(
                "CUDA requires AWQ GEMM W4A16 with packed zero points: G{} {:?}",
                format.group_size, format.packing
            ),
        ),
        (BackendTarget::Metal, WeightEncoding::Gptq { format }) => gptq("Metal", *format, true),
        (BackendTarget::Cuda, WeightEncoding::Gptq { format }) => gptq("CUDA", *format, true),
        (
            BackendTarget::Metal | BackendTarget::Cuda,
            WeightEncoding::BitsAndBytes4Bit { format },
        ) => bitsandbytes(*format),
        (BackendTarget::Metal, WeightEncoding::Float8 { format }) => float8_metal(*format),
        (BackendTarget::Cuda, WeightEncoding::Float8 { format }) => float8_cuda(*format),
        (BackendTarget::Metal | BackendTarget::Cuda, WeightEncoding::MxFp4 { .. }) => (
            AdmissionCheckKind::MxFp4,
            AdmissionStatus::Partial,
            "MXFP4 is implemented only for selected model compositions".into(),
        ),
        (BackendTarget::Metal, WeightEncoding::MxFp8 { format }) => (
            AdmissionCheckKind::MxFp8,
            if *format == models::weights::BlockQuantization::MXFP8 {
                AdmissionStatus::Partial
            } else {
                AdmissionStatus::Unsupported
            },
            "Metal supports native weight-only MXFP8 for separate dense projections".into(),
        ),
        (BackendTarget::Cuda, WeightEncoding::MxFp8 { format }) => (
            AdmissionCheckKind::MxFp8,
            if *format == models::weights::BlockQuantization::MXFP8 {
                AdmissionStatus::Partial
            } else {
                AdmissionStatus::Unsupported
            },
            "CUDA supports native weight-only MXFP8 for separate dense projections".into(),
        ),
        (BackendTarget::Metal, WeightEncoding::NvFp4 { format }) => (
            AdmissionCheckKind::NvFp4,
            if *format == models::weights::BlockQuantization::NVFP4 {
                AdmissionStatus::Partial
            } else {
                AdmissionStatus::Unsupported
            },
            "Metal supports ModelOpt NVFP4 through role-specific dense conversion and direct \
             routed execution"
                .into(),
        ),
        (BackendTarget::Cuda, WeightEncoding::NvFp4 { .. }) => (
            AdmissionCheckKind::NvFp4,
            AdmissionStatus::Partial,
            "CUDA supports NVFP4 in admitted dense and routed-MoE compositions".into(),
        ),
    };
    AdmissionCheck { kind, status, detail }
}

fn gptq(
    backend: &str,
    format: GptqQuantization,
    allow_activation_order: bool,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let admitted = format.bits == GptqBits::Four
        && format.scale_dtype == GptqScaleDType::F16
        && format.symmetric
        && (allow_activation_order || !format.activation_order)
        && format.is_input_packed();
    let status = if admitted {
        AdmissionStatus::Partial
    } else {
        AdmissionStatus::Unsupported
    };
    (
        AdmissionCheckKind::Gptq,
        status,
        format!(
            "{backend} requires symmetric GPTQ W4A16 input packing; activation ordering: {}: {:?} G{}",
            if allow_activation_order {
                "supported"
            } else {
                "unsupported"
            },
            format.checkpoint_format,
            format.group_size
        ),
    )
}

fn packed_int8(
    backend: &str,
    format: CompressedIntegerQuantization,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let status = if format.is_symmetric_channel_int8()
        && format.scale_dtype == CompressedIntegerScaleDType::BF16
    {
        AdmissionStatus::Partial
    } else {
        AdmissionStatus::Unsupported
    };
    (
        AdmissionCheckKind::PackedInt8,
        status,
        format!(
            "{backend} requires symmetric per-channel offset-binary INT8 in dense I32 packing with \
             BF16 scales"
        ),
    )
}

fn packed_int4(
    backend: &str,
    format: CompressedIntegerQuantization,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let status = if format.is_symmetric_group_int4()
        && format.scale_dtype == CompressedIntegerScaleDType::BF16
    {
        AdmissionStatus::Partial
    } else {
        AdmissionStatus::Unsupported
    };
    (
        AdmissionCheckKind::PackedInt4,
        status,
        format!(
            "{backend} requires symmetric grouped offset-binary INT4 in dense I32 packing with \
             BF16 scales"
        ),
    )
}

fn dense(dtype: &str, detail: &str) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let status = if matches!(dtype.to_ascii_uppercase().as_str(), "BF16" | "F16" | "F32") {
        AdmissionStatus::Partial
    } else {
        AdmissionStatus::Unsupported
    };
    (AdmissionCheckKind::Dense, status, detail.into())
}

fn affine_metal(
    format: GroupedAffineQuantization,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let native = native_mlx(format)
        && format.storage_dtype == AffineStorageDType::U32
        && matches!(
            format.scale_dtype,
            AffineParameterDType::F16 | AffineParameterDType::BF16 | AffineParameterDType::F32
        );
    let status = if native {
        AdmissionStatus::Partial
    } else {
        AdmissionStatus::Unsupported
    };
    (
        AdmissionCheckKind::Affine,
        status,
        "Metal requires unsigned input-grouped MLX U32 packing with additive biases".into(),
    )
}

fn affine_cuda(format: GroupedAffineQuantization) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let native = native_mlx(format)
        && format.storage_dtype == AffineStorageDType::U32
        && format.scale_dtype == AffineParameterDType::BF16;
    let status = if native {
        AdmissionStatus::Partial
    } else {
        AdmissionStatus::Unsupported
    };
    (
        AdmissionCheckKind::Affine,
        status,
        "CUDA requires BF16-parameter unsigned MLX affine U32 packing with additive biases".into(),
    )
}

fn native_mlx(format: GroupedAffineQuantization) -> bool {
    format.is_mlx_layout() && format.has_additive_bias()
}
