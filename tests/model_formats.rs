use foundation::model::BackendTarget;
use libmir::{
    BackendAdmissionReport, CheckpointEncoding, MODEL_FORMAT_REGISTRY_SCHEMA_VERSION,
    WeightEncoding,
};
use models::weights::{
    AffineBits, AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness,
    AffineStorageDType, AffineZeroPointMode, AwqBits, AwqPacking, AwqQuantization, AwqScaleDType,
    AwqStorageDType, BlockQuantization, CompressedIntegerActivationOrder, CompressedIntegerBits,
    CompressedIntegerPacking, CompressedIntegerQuantization, CompressedIntegerScaleDType,
    CompressedIntegerScaleStrategy, CompressedIntegerSignedness, CompressedIntegerStorageDType,
    CompressedIntegerZeroPointMode, Float8ActivationScale, Float8Format, Float8ParameterDType,
    Float8Quantization, Float8ScaleGranularity, Float8ScaleMode, GptqBits, GptqCheckpointFormat,
    GptqPacking, GptqQuantization, GptqScaleDType, GptqStorageDType, GroupedAffineQuantization,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Matrix {
    schema_version: u32,
    format: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    id: String,
    metal: String,
    cuda: String,
}

#[test]
fn checked_in_format_matrix_matches_the_capability_registry()
-> Result<(), Box<dyn std::error::Error>> {
    let matrix: Matrix = toml::from_str(include_str!("../validation/model-formats.toml"))?;
    assert_eq!(matrix.schema_version, MODEL_FORMAT_REGISTRY_SCHEMA_VERSION);
    for row in matrix.format {
        let Some(weight) = encoding(&row.id) else {
            return Err(
                std::io::Error::other(format!("unknown validation format {}", row.id)).into()
            );
        };
        let encoding = CheckpointEncoding { weights: vec![weight] };
        let metal = BackendAdmissionReport::for_encoding(BackendTarget::Metal, encoding.clone());
        let cuda = BackendAdmissionReport::for_encoding(BackendTarget::Cuda, encoding);
        assert_eq!(metal.status.as_str(), row.metal, "Metal row {}", row.id);
        assert_eq!(cuda.status.as_str(), row.cuda, "CUDA row {}", row.id);
    }
    Ok(())
}

#[test]
fn gptq_v1_and_v2_share_admission_but_not_serialized_zero_semantics() {
    for checkpoint_format in [GptqCheckpointFormat::Gptq, GptqCheckpointFormat::GptqV2] {
        let mut format = gptq();
        format.checkpoint_format = checkpoint_format;
        let encoding = CheckpointEncoding {
            weights: vec![WeightEncoding::Gptq { format }],
        };
        for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
            assert_eq!(
                BackendAdmissionReport::for_encoding(backend, encoding.clone()).status.as_str(),
                "partial"
            );
        }
    }
}

#[test]
fn gptq_activation_order_is_admitted_on_both_accelerators() {
    let mut format = gptq();
    format.activation_order = true;
    let encoding = CheckpointEncoding {
        weights: vec![WeightEncoding::Gptq { format }],
    };
    assert_eq!(
        BackendAdmissionReport::for_encoding(BackendTarget::Metal, encoding.clone())
            .status
            .as_str(),
        "partial"
    );
    assert_eq!(
        BackendAdmissionReport::for_encoding(BackendTarget::Cuda, encoding)
            .status
            .as_str(),
        "partial"
    );
}

fn encoding(id: &str) -> Option<WeightEncoding> {
    Some(match id {
        "dense_bf16" => WeightEncoding::Dense { dtype: "BF16".into() },
        "dense_f16" => WeightEncoding::Dense { dtype: "F16".into() },
        "dense_f32" => WeightEncoding::Dense { dtype: "F32".into() },
        "packed_int8" => WeightEncoding::PackedInt8 { format: packed_int8() },
        "packed_int4" => WeightEncoding::PackedInt4 { format: packed_int4() },
        "awq" => WeightEncoding::Awq { format: awq() },
        "gptq" => WeightEncoding::Gptq { format: gptq() },
        "fp8_e4m3" => WeightEncoding::Float8 {
            format: Float8Quantization {
                format: Float8Format::E4M3,
                scale_mode: Float8ScaleMode::Multiplier,
                scale_granularity: Float8ScaleGranularity::OutputChannel,
                scale_dtype: Some(Float8ParameterDType::BF16),
                activation_scale: Float8ActivationScale::DynamicToken,
                input_scale_dtype: None,
            },
        },
        "fp8_e5m2" => WeightEncoding::Float8 {
            format: Float8Quantization {
                format: Float8Format::E5M2,
                scale_mode: Float8ScaleMode::Multiplier,
                scale_granularity: Float8ScaleGranularity::Tensor,
                scale_dtype: Some(Float8ParameterDType::F32),
                activation_scale: Float8ActivationScale::None,
                input_scale_dtype: None,
            },
        },
        "mxfp4" => WeightEncoding::MxFp4 { format: BlockQuantization::MXFP4 },
        "mxfp8" => WeightEncoding::MxFp8 { format: BlockQuantization::MXFP8 },
        "nvfp4" => WeightEncoding::NvFp4 { format: BlockQuantization::NVFP4 },
        value if value.starts_with("affine_") => WeightEncoding::Affine {
            format: affine(value.strip_prefix("affine_")?.parse().ok()?)?,
        },
        _ => return None,
    })
}

fn awq() -> AwqQuantization {
    AwqQuantization {
        bits: AwqBits::Four,
        group_size: 128,
        packing: AwqPacking::GemmOutputInterleaved,
        storage_dtype: AwqStorageDType::I32,
        scale_dtype: AwqScaleDType::F16,
        packed_zero_points: true,
    }
}

fn gptq() -> GptqQuantization {
    GptqQuantization {
        bits: GptqBits::Four,
        group_size: 128,
        packing: GptqPacking::InputLittleEndian,
        storage_dtype: GptqStorageDType::I32,
        scale_dtype: GptqScaleDType::F16,
        checkpoint_format: GptqCheckpointFormat::Gptq,
        symmetric: true,
        activation_order: false,
        packed_zero_points: true,
    }
}

fn packed_int8() -> CompressedIntegerQuantization {
    CompressedIntegerQuantization {
        bits: CompressedIntegerBits::Eight,
        scale_strategy: CompressedIntegerScaleStrategy::Channel,
        signedness: CompressedIntegerSignedness::OffsetBinary,
        zero_point: CompressedIntegerZeroPointMode::None,
        activation_order: CompressedIntegerActivationOrder::None,
        packing: CompressedIntegerPacking::DenseLittleEndian,
        storage_dtype: CompressedIntegerStorageDType::I32,
        scale_dtype: CompressedIntegerScaleDType::BF16,
    }
}

fn packed_int4() -> CompressedIntegerQuantization {
    CompressedIntegerQuantization {
        bits: CompressedIntegerBits::Four,
        scale_strategy: CompressedIntegerScaleStrategy::Group { group_size: 128 },
        signedness: CompressedIntegerSignedness::OffsetBinary,
        zero_point: CompressedIntegerZeroPointMode::None,
        activation_order: CompressedIntegerActivationOrder::None,
        packing: CompressedIntegerPacking::DenseLittleEndian,
        storage_dtype: CompressedIntegerStorageDType::I32,
        scale_dtype: CompressedIntegerScaleDType::BF16,
    }
}

fn affine(bits: u8) -> Option<GroupedAffineQuantization> {
    Some(GroupedAffineQuantization {
        bits: AffineBits::try_from(bits).ok()?,
        group_size: 64,
        group_axis: AffineGroupAxis::Input,
        signedness: AffineSignedness::Unsigned,
        zero_point: AffineZeroPointMode::AdditiveBias,
        packing: AffinePacking::Mlx,
        storage_dtype: AffineStorageDType::U32,
        scale_dtype: AffineParameterDType::BF16,
        bias_dtype: Some(AffineParameterDType::BF16),
    })
}
