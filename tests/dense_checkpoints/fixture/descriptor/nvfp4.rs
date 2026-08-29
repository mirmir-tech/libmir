use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ModelDescriptor, WeightEncoding};
use models::weights::{
    BlockProjectionLayout, BlockQuantization, ExpertProjectionRole, LayerTensorRole,
    LogicalTensorRole,
};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_nvfp4_descriptor(
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
    require(super::family(descriptor) == expected_family, "semantic family differs")?;
    super::validate_tokenizer(descriptor, reference)?;
    validate_storage(descriptor, reference)?;
    validate_individual_experts(descriptor)?;
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected NVFP4 storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let mut dense = BTreeSet::new();
    let mut nvfp4 = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::NvFp4 { format } => {
                require(format == BlockQuantization::NVFP4, "checkpoint NVFP4 format differs")?;
                nvfp4.insert(format);
            },
            _ => return Err(validation_error("NVFP4 fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    require(nvfp4 == BTreeSet::from([BlockQuantization::NVFP4]), "NVFP4 format is absent")
}

fn validate_individual_experts(descriptor: &ModelDescriptor) -> TestResult<()> {
    let bindings = &descriptor
        .execution()
        .ok_or_else(|| validation_error("checkpoint has no execution contract"))?
        .bindings;
    let mut gate = BTreeSet::new();
    let mut up = BTreeSet::new();
    let mut down = BTreeSet::new();
    for binding in &bindings.tensors {
        let LogicalTensorRole::Layer {
            tensor: LayerTensorRole::ExpertProjection { expert: Some(expert), projection },
            ..
        } = binding.role
        else {
            continue;
        };
        require(
            binding.block_projection_layout() == Some(BlockProjectionLayout::Matrix),
            "NVFP4 individual expert is not an ordinary packed matrix",
        )?;
        match projection {
            ExpertProjectionRole::Gate => gate.insert(expert),
            ExpertProjectionRole::Up => up.insert(expert),
            ExpertProjectionRole::Down => down.insert(expert),
            ExpertProjectionRole::GateUp => false,
        };
    }
    require(!gate.is_empty() && gate == up && gate == down, "NVFP4 expert sets differ")
}
