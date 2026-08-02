use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ModelDescriptor, WeightEncoding};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_bitsandbytes_descriptor(
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
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected bitsandbytes storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let mut dense = BTreeSet::new();
    let mut bitsandbytes = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::BitsAndBytes4Bit { format } => {
                bitsandbytes.insert(format);
            },
            _ => {
                return Err(validation_error(
                    "bitsandbytes fixture contains another packed encoding",
                ));
            },
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    let expected = reference
        .bitsandbytes_4bit
        .as_ref()
        .ok_or_else(|| validation_error("reference has no bitsandbytes contract"))?
        .format();
    require(
        bitsandbytes == BTreeSet::from([expected]),
        format!("bitsandbytes formats differ: actual={bitsandbytes:?}, expected={expected:?}"),
    )
}
