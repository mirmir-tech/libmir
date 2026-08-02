use std::collections::BTreeSet;

use libmir::{
    AdmissionStatus, ModelDescriptor, WeightEncoding, models::weights::BlockQuantization,
};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_mxfp8_descriptor(
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
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected MXFP8 storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let expected = reference
        .mxfp8
        .as_ref()
        .ok_or_else(|| validation_error("MXFP8 reference has no storage contract"))?;
    let mut dense = BTreeSet::new();
    let mut mxfp8 = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::MxFp8 { format } => {
                require(format == BlockQuantization::MXFP8, "checkpoint MXFP8 format differs")?;
                mxfp8.insert((
                    format.block_size,
                    format.storage_dtype.as_str(),
                    format.block_scale.encoding.as_str(),
                    format.block_scale.storage_dtype.as_str(),
                ));
            },
            _ => return Err(validation_error("MF-130 fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    require(
        mxfp8
            == BTreeSet::from([(
                expected.block_size,
                expected.storage_dtype.as_str(),
                expected.scale_encoding.as_str(),
                expected.scale_dtype.as_str(),
            )]),
        format!("MXFP8 formats differ from reference: {mxfp8:?}"),
    )
}
