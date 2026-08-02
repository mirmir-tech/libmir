use models::weights::{
    BindingTransform, CompressedIntegerActivationOrder, CompressedIntegerBits,
    CompressedIntegerPacking, CompressedIntegerQuantization, CompressedIntegerScaleDType,
    CompressedIntegerScaleStrategy, CompressedIntegerSignedness, CompressedIntegerStorageDType,
    CompressedIntegerZeroPointMode, LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::{packed_shape_source, runtime_raw_sources};

#[test]
fn packed_int8_boundary_upload_excludes_shape_metadata() {
    let binding = TensorBinding {
        role: LogicalTensorRole::Embedding,
        source: "model.embed_tokens.weight_packed".into(),
        shape: vec![2, 4],
        logical_shape: Some(vec![2, 16]),
        transforms: Vec::<BindingTransform>::new(),
        storage: TensorStorage::PackedInt8 {
            format: CompressedIntegerQuantization {
                bits: CompressedIntegerBits::Eight,
                scale_strategy: CompressedIntegerScaleStrategy::Channel,
                signedness: CompressedIntegerSignedness::OffsetBinary,
                zero_point: CompressedIntegerZeroPointMode::None,
                activation_order: CompressedIntegerActivationOrder::None,
                packing: CompressedIntegerPacking::DenseLittleEndian,
                storage_dtype: CompressedIntegerStorageDType::I32,
                scale_dtype: CompressedIntegerScaleDType::BF16,
            },
            scales: "model.embed_tokens.weight_scale".into(),
            shape: "model.embed_tokens.weight_shape".into(),
            zero_points: None,
            group_indices: None,
        },
    };

    assert_eq!(
        runtime_raw_sources(&binding),
        ["model.embed_tokens.weight_packed", "model.embed_tokens.weight_scale"]
    );
    assert_eq!(packed_shape_source(&binding), Some("model.embed_tokens.weight_shape"));
}
