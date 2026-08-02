use foundation::model::BackendTarget;
use models::weights::{
    AffineBits, AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness,
    AffineStorageDType, AffineZeroPointMode, BlockQuantization, CompressedIntegerActivationOrder,
    CompressedIntegerBits, CompressedIntegerPacking, CompressedIntegerQuantization,
    CompressedIntegerScaleDType, CompressedIntegerScaleStrategy, CompressedIntegerSignedness,
    CompressedIntegerStorageDType, CompressedIntegerZeroPointMode, GroupedAffineQuantization,
    LogicalTensorRole, TensorBinding, TensorStorage, WeightBindingPlan,
};

use super::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, CheckpointEncoding, WeightEncoding,
    aggregate, registry, resolve_dense_check,
};

mod formats;

#[test]
fn checkpoint_encoding_is_derived_from_weight_bindings() {
    let bindings = WeightBindingPlan {
        tensors: vec![
            binding(
                LogicalTensorRole::Embedding,
                TensorStorage::Dense { dtype: "BF16".into(), bias: None },
            ),
            binding(
                LogicalTensorRole::Output,
                TensorStorage::Dense { dtype: "BF16".into(), bias: None },
            ),
            binding(
                LogicalTensorRole::Auxiliary { path: "packed.weight".into() },
                TensorStorage::AffineQuantized {
                    format: affine(AffineBits::Four),
                    scales: "packed.scales".into(),
                    biases: Some("packed.biases".into()),
                    output_bias: None,
                },
            ),
            binding(
                LogicalTensorRole::Auxiliary { path: "packed3.weight".into() },
                TensorStorage::AffineQuantized {
                    format: affine(AffineBits::Three),
                    scales: "packed3.scales".into(),
                    biases: Some("packed3.biases".into()),
                    output_bias: None,
                },
            ),
            binding(
                LogicalTensorRole::Auxiliary { path: "rope.freqs".into() },
                TensorStorage::Auxiliary { dtype: "F32".into() },
            ),
        ],
    };

    assert_eq!(
        CheckpointEncoding::from_bindings(&bindings).weights,
        vec![
            WeightEncoding::Dense { dtype: "BF16".into() },
            WeightEncoding::Affine { format: affine(AffineBits::Three) },
            WeightEncoding::Affine { format: affine(AffineBits::Four) },
        ]
    );
}

#[test]
fn affine_admission_uses_the_complete_physical_contract() {
    let mut no_bias = affine(AffineBits::Four);
    no_bias.zero_point = AffineZeroPointMode::None;
    no_bias.bias_dtype = None;
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &WeightEncoding::Affine { format: no_bias }).status,
        AdmissionStatus::Unsupported
    );

    let mut signed = affine(AffineBits::Four);
    signed.signedness = AffineSignedness::Signed;
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &WeightEncoding::Affine { format: signed }).status,
        AdmissionStatus::Unsupported
    );

    let mut f16 = affine(AffineBits::Four);
    f16.scale_dtype = AffineParameterDType::F16;
    f16.bias_dtype = Some(AffineParameterDType::F16);
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &WeightEncoding::Affine { format: f16 }).status,
        AdmissionStatus::Partial
    );
    assert_eq!(
        registry::assess(&BackendTarget::Cuda, &WeightEncoding::Affine { format: f16 }).status,
        AdmissionStatus::Unsupported
    );

    let mut byte_storage = affine(AffineBits::Eight);
    byte_storage.storage_dtype = AffineStorageDType::U8;
    assert_eq!(
        registry::assess(&BackendTarget::Metal, &WeightEncoding::Affine { format: byte_storage })
            .status,
        AdmissionStatus::Unsupported
    );
}

#[test]
fn packed_int8_admission_requires_native_bf16_scales() {
    let mut format = packed_int8();
    format.scale_dtype = CompressedIntegerScaleDType::F16;
    for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
        assert_eq!(
            registry::assess(&backend, &WeightEncoding::PackedInt8 { format }).status,
            AdmissionStatus::Unsupported
        );
    }
    format.scale_dtype = CompressedIntegerScaleDType::F32;
    for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
        assert_eq!(
            registry::assess(&backend, &WeightEncoding::PackedInt8 { format }).status,
            AdmissionStatus::Unsupported
        );
    }
}

#[test]
fn unsupported_requirement_dominates_report() {
    let supported =
        registry::assess(&BackendTarget::Metal, &WeightEncoding::Dense { dtype: "BF16".into() });
    let unsupported = registry::assess(
        &BackendTarget::Metal,
        &WeightEncoding::NvFp4 { format: BlockQuantization::MXFP8 },
    );
    assert_eq!(aggregate(&[supported, unsupported]), AdmissionStatus::Unsupported);
}

#[test]
fn empty_contract_stays_unknown() {
    assert_eq!(aggregate(&[]), AdmissionStatus::Unknown);
}

#[test]
fn resolved_dense_execution_replaces_conditional_format_status() {
    let mut checks = vec![registry::assess(
        &BackendTarget::Metal,
        &WeightEncoding::Dense { dtype: "BF16".into() },
    )];
    resolve_dense_check(
        &mut checks,
        AdmissionCheck {
            kind: AdmissionCheckKind::Dense,
            status: AdmissionStatus::Supported,
            detail: "dense execution is validated".into(),
        },
    );

    assert_eq!(checks.len(), 1);
    assert_eq!(aggregate(&checks), AdmissionStatus::Supported);
}

fn binding(role: LogicalTensorRole, storage: TensorStorage) -> TensorBinding {
    TensorBinding {
        role,
        source: "weight".into(),
        shape: vec![64, 64],
        logical_shape: None,
        transforms: Vec::new(),
        storage,
    }
}

fn affine(bits: AffineBits) -> GroupedAffineQuantization {
    GroupedAffineQuantization {
        bits,
        group_size: 64,
        group_axis: AffineGroupAxis::Input,
        signedness: AffineSignedness::Unsigned,
        zero_point: AffineZeroPointMode::AdditiveBias,
        packing: AffinePacking::Mlx,
        storage_dtype: AffineStorageDType::U32,
        scale_dtype: AffineParameterDType::BF16,
        bias_dtype: Some(AffineParameterDType::BF16),
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
