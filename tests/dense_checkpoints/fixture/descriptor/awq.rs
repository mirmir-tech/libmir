use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ModelDescriptor, WeightEncoding};

use super::{family, validate_tokenizer};
use crate::{
    TestResult,
    fixture::{Family, Reference, active_target, require, validation_error},
};

pub fn validate_awq_descriptor(
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
    require(family(descriptor) == expected_family, "semantic family differs from fixture")?;
    validate_tokenizer(descriptor, reference)?;
    validate_storage(descriptor, reference)?;
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected AWQ storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let expected = reference
        .awq
        .as_ref()
        .ok_or_else(|| validation_error("AWQ reference has no format contract"))?;
    let mut dense = BTreeSet::new();
    let mut awq = Vec::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::Awq { format } => awq.push(format),
            _ => return Err(validation_error("AWQ fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    require(
        awq.len() == 1
            && awq[0].is_gemm_w4a16()
            && awq[0].bits.get() == expected.bits
            && awq[0].group_size == expected.group_size,
        "AWQ format differs from reference",
    )
}
