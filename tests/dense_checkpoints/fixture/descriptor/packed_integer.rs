use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ModelDescriptor, WeightEncoding};
use models::weights::{CompressedIntegerScaleDType, CompressedIntegerScaleStrategy};

use super::{family, validate_tokenizer};
use crate::{
    TestResult,
    fixture::{Family, Reference, active_target, require, validation_error},
};

pub fn validate_packed_int8_descriptor(
    descriptor: &ModelDescriptor,
    expected_family: Family,
    reference: &Reference,
) -> TestResult<()> {
    validate_packed_descriptor(descriptor, expected_family, reference, 8)
}

pub fn validate_packed_int4_descriptor(
    descriptor: &ModelDescriptor,
    expected_family: Family,
    reference: &Reference,
) -> TestResult<()> {
    validate_packed_descriptor(descriptor, expected_family, reference, 4)
}

fn validate_packed_descriptor(
    descriptor: &ModelDescriptor,
    expected_family: Family,
    reference: &Reference,
    bits: u8,
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
    validate_packed_integer_storage(descriptor, reference, bits)?;
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected packed INT{bits} storage: {admission:?}"),
    )
}

fn validate_packed_integer_storage(
    descriptor: &ModelDescriptor,
    reference: &Reference,
    bits: u8,
) -> TestResult<()> {
    let expected = match bits {
        4 => reference.packed_int4.as_ref(),
        8 => reference.packed_int8.as_ref(),
        _ => None,
    }
    .ok_or_else(|| validation_error("reference has no packed integer contract"))?;
    let mut dense = BTreeSet::new();
    let mut packed = Vec::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::PackedInt8 { format } if bits == 8 => packed.push(format),
            WeightEncoding::PackedInt4 { format } if bits == 4 => packed.push(format),
            _ => return Err(validation_error("MF-120 fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    let contract = packed.first().is_some_and(|format| {
        let geometry = if bits == 4 {
            format.is_symmetric_group_int4()
                && matches!(
                    format.scale_strategy,
                    CompressedIntegerScaleStrategy::Group { group_size }
                        if Some(group_size) == expected.group_size
                )
        } else {
            format.is_symmetric_channel_int8()
        };
        geometry
            && format.bits.get() == expected.bits
            && format.scale_dtype == CompressedIntegerScaleDType::BF16
    });
    require(packed.len() == 1 && contract, "packed integer format differs from reference")
}
