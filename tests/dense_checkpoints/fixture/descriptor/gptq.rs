use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ModelDescriptor, WeightEncoding};
use models::weights::GptqCheckpointFormat;

use super::{family, validate_tokenizer};
use crate::{
    TestResult,
    fixture::{Family, Reference, active_target, require, validation_error},
};

pub fn validate_gptq_descriptor(
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
        format!("active backend rejected GPTQ storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let expected = reference
        .gptq
        .as_ref()
        .ok_or_else(|| validation_error("GPTQ reference has no format contract"))?;
    let expected_checkpoint = match expected.checkpoint_format.as_str() {
        "gptq" => GptqCheckpointFormat::Gptq,
        "gptq_v2" => GptqCheckpointFormat::GptqV2,
        _ => return Err(validation_error("GPTQ reference has an unknown checkpoint format")),
    };
    let mut dense = BTreeSet::new();
    let mut gptq = Vec::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::Gptq { format } => gptq.push(format),
            _ => return Err(validation_error("GPTQ fixture contains another packed encoding")),
        }
    }
    require(
        dense == reference.dtypes.iter().cloned().collect(),
        "dense companion dtypes differ from reference",
    )?;
    require(
        gptq.len() == 1
            && gptq[0].bits.get() == expected.bits
            && gptq[0].group_size == expected.group_size
            && gptq[0].checkpoint_format == expected_checkpoint
            && gptq[0].symmetric == expected.symmetric
            && gptq[0].activation_order == expected.activation_order
            && gptq[0].scale_dtype.as_str() == expected.scale_dtype
            && gptq[0].is_input_packed(),
        "GPTQ format differs from reference",
    )
}
