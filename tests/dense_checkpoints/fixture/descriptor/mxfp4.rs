use std::collections::BTreeSet;

use libmir::{
    AdmissionStatus, ModelDescriptor, WeightEncoding,
    models::weights::{
        BlockProjectionLayout, BlockStorageDType, ExpertProjectionRole, LayerTensorRole,
        LogicalTensorRole,
    },
};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_mxfp4_descriptor(
    descriptor: &ModelDescriptor,
    expected_family: Family,
    reference: &Reference,
) -> TestResult<()> {
    require(
        descriptor.metadata().model_type == reference.model_type,
        "checkpoint model_type differs from reference",
    )?;
    require(
        descriptor.metadata().context_len == reference.context_len,
        "checkpoint context length differs from reference",
    )?;
    let execution = descriptor
        .execution()
        .ok_or_else(|| validation_error("checkpoint has no generation execution contract"))?;
    require(
        execution.semantic.decoder.vocab_size == reference.vocab_size,
        "vocabulary differs",
    )?;
    require(
        super::family(descriptor) == expected_family,
        "semantic family differs from fixture",
    )?;
    super::validate_tokenizer(descriptor, reference)?;
    validate_storage(descriptor, reference)?;
    validate_routed_layout(descriptor, reference)?;
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected MXFP4 storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let expected = reference
        .mxfp4
        .as_ref()
        .ok_or_else(|| validation_error("MXFP4 reference has no storage contract"))?;
    let mut dense = BTreeSet::new();
    let mut affine = BTreeSet::new();
    let mut mxfp4 = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::MxFp4 { format } => {
                require(format.is_mxfp4(), "checkpoint MXFP4 format differs")?;
                mxfp4.insert((
                    format.block_size,
                    format.storage_dtype.as_str(),
                    format.block_scale.encoding.as_str(),
                    format.block_scale.storage_dtype.as_str(),
                    format.output_bias_dtype.map(BlockStorageDType::as_str),
                ));
            },
            WeightEncoding::Affine { format } => {
                require(super::native_mlx(format), "checkpoint affine companion is not MLX")?;
                affine.insert((
                    format.bits.get(),
                    format.group_size,
                    format.scale_dtype.as_str().to_owned(),
                ));
            },
            _ => return Err(validation_error("MF-130 fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    let expected_affine = reference.affine.as_ref().map_or_else(BTreeSet::new, |format| {
        format
            .bits
            .iter()
            .flat_map(|bits| {
                format
                    .group_sizes
                    .iter()
                    .map(|group| (*bits, *group, format.parameter_dtype.clone()))
            })
            .collect()
    });
    require(affine == expected_affine, "affine companion formats differ from reference")?;
    require(
        mxfp4
            == BTreeSet::from([(
                expected.block_size,
                expected.storage_dtype.as_str(),
                expected.scale_encoding.as_str(),
                expected.scale_dtype.as_str(),
                Some(expected.output_bias_dtype.as_str()),
            )]),
        format!("MXFP4 formats differ from reference: {mxfp4:?}"),
    )
}

fn validate_routed_layout(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let bindings = &descriptor
        .execution()
        .ok_or_else(|| validation_error("checkpoint has no execution contract"))?
        .bindings;
    let mut gate_up_counts = BTreeSet::new();
    let mut gate_counts = BTreeSet::new();
    let mut up_counts = BTreeSet::new();
    let mut down_counts = BTreeSet::new();
    for binding in &bindings.tensors {
        let role = match binding.role {
            LogicalTensorRole::Layer {
                tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
                ..
            } => Some(projection),
            _ => None,
        };
        match (role, binding.block_projection_layout()) {
            (
                Some(ExpertProjectionRole::GateUp),
                Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true }),
            ) => {
                gate_up_counts.insert(experts);
            },
            (
                Some(ExpertProjectionRole::Gate),
                Some(BlockProjectionLayout::MatrixBank { matrices }),
            ) => {
                gate_counts.insert(matrices);
            },
            (
                Some(ExpertProjectionRole::Up),
                Some(BlockProjectionLayout::MatrixBank { matrices }),
            ) => {
                up_counts.insert(matrices);
            },
            (
                Some(ExpertProjectionRole::Down),
                Some(BlockProjectionLayout::MatrixBank { matrices }),
            ) => {
                down_counts.insert(matrices);
            },
            (Some(_), _) => {
                return Err(validation_error("MXFP4 expert layout is not routed-native"));
            },
            _ => {},
        }
    }
    let layout = &reference
        .mxfp4
        .as_ref()
        .ok_or_else(|| validation_error("MXFP4 reference has no format contract"))?
        .routed_layout;
    let valid = match layout.as_str() {
        "interleaved_gate_up_bank" => {
            gate_counts.is_empty()
                && up_counts.is_empty()
                && !gate_up_counts.is_empty()
                && gate_up_counts == down_counts
        },
        "separate_gate_up_banks" => {
            gate_up_counts.is_empty()
                && !gate_counts.is_empty()
                && gate_counts == up_counts
                && gate_counts == down_counts
        },
        _ => false,
    };
    require(valid, "MXFP4 routed matrix-bank layout differs from reference")
}
