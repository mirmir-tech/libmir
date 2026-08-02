use std::collections::BTreeSet;

use libmir::{
    AdmissionStatus, ModelDescriptor, WeightEncoding,
    models::weights::{
        Float8ActivationScale, Float8ParameterDType, Float8ScaleGranularity, Float8ScaleMode,
    },
};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_float8_descriptor(
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
        format!("active backend rejected FP8 storage: {admission:?}"),
    )
}

fn validate_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let expected = reference
        .float8
        .as_ref()
        .ok_or_else(|| validation_error("FP8 reference has no storage contract"))?;
    let mut dense = BTreeSet::new();
    let mut float8 = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::Float8 { format } => {
                require(
                    format.format.as_str() == expected.format,
                    "checkpoint FP8 value format differs from reference",
                )?;
                require(
                    matches!(format.scale_mode, Float8ScaleMode::None)
                        || matches!(
                            format.scale_mode,
                            Float8ScaleMode::Multiplier | Float8ScaleMode::InverseMultiplier
                        ),
                    "checkpoint FP8 scale mode is unsupported",
                )?;
                require(
                    matches!(format.scale_granularity, Float8ScaleGranularity::None)
                        || matches!(
                            format.scale_granularity,
                            Float8ScaleGranularity::Tensor | Float8ScaleGranularity::OutputChannel
                        ),
                    "checkpoint FP8 scale granularity is unsupported",
                )?;
                float8.insert((
                    format.format.as_str(),
                    format.scale_mode.as_str(),
                    format.scale_granularity.as_str(),
                    format.scale_dtype.map(Float8ParameterDType::as_str),
                    activation_scale(format.activation_scale),
                    format.input_scale_dtype.map(Float8ParameterDType::as_str),
                ));
            },
            _ => return Err(validation_error("MF-130 fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    require(
        float8
            == BTreeSet::from([(
                expected.format.as_str(),
                expected.scale_mode.as_str(),
                expected.scale_granularity.as_str(),
                expected.scale_dtype.as_deref(),
                expected.activation_scale.as_deref(),
                expected.input_scale_dtype.as_deref(),
            )]),
        format!("FP8 formats differ from reference: {float8:?}"),
    )
}

fn activation_scale(scale: Float8ActivationScale) -> Option<&'static str> {
    match scale {
        Float8ActivationScale::None => None,
        Float8ActivationScale::StaticTensor => Some("static_tensor"),
        Float8ActivationScale::DynamicToken => Some("dynamic_token"),
    }
}
