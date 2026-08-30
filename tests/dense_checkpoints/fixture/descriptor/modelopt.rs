use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ModelDescriptor, WeightEncoding};
use models::weights::{BlockQuantization, Float8ActivationScale};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_modelopt_descriptor(
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
        format!("active backend rejected mixed ModelOpt storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let mut dense = BTreeSet::new();
    let mut float8 = false;
    let mut nvfp4 = false;
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::Float8 { format } => {
                float8 |= format.format.as_str() == "F8_E4M3"
                    && format.scale_mode.as_str() == "multiplier"
                    && format.scale_granularity.as_str() == "tensor"
                    && format.scale_dtype.is_some_and(|dtype| dtype.as_str() == "F32")
                    && format.activation_scale == Float8ActivationScale::StaticTensor
                    && format.input_scale_dtype.is_some_and(|dtype| dtype.as_str() == "F32");
            },
            WeightEncoding::NvFp4 { format } => {
                nvfp4 |= format == BlockQuantization::NVFP4_W4A16;
            },
            _ => return Err(validation_error("mixed ModelOpt checkpoint has another encoding")),
        }
    }
    let expected = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected, format!("dense dtypes differ: {dense:?}"))?;
    require(float8, "ModelOpt FP8 encoding is absent or unsupported")?;
    require(nvfp4, "ModelOpt NVFP4 encoding is absent or unsupported")
}
