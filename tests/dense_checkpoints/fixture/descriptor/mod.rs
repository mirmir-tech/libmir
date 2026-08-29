use std::collections::BTreeSet;

use libmir::{AdmissionStatus, ArchitectureCapability, ModelDescriptor, WeightEncoding};
use models::weights::{AffineStorageDType, GroupedAffineQuantization};

use super::{Family, Reference, TestResult, active_target, require, validation_error};

mod awq;
mod bitsandbytes;
mod float8;
mod gptq;
mod modelopt;
mod mxfp4;
mod mxfp8;
mod nvfp4;
mod packed_integer;
pub use awq::validate_awq_descriptor;
pub use bitsandbytes::validate_bitsandbytes_descriptor;
pub use float8::validate_float8_descriptor;
pub use gptq::validate_gptq_descriptor;
pub use modelopt::validate_modelopt_descriptor;
pub use mxfp4::validate_mxfp4_descriptor;
pub use mxfp8::validate_mxfp8_descriptor;
pub use nvfp4::validate_nvfp4_descriptor;
pub use packed_integer::{validate_packed_int4_descriptor, validate_packed_int8_descriptor};

pub fn validate_descriptor(
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
    validate_dense_storage(descriptor, reference)?;
    let admission = descriptor.admission(active_target());
    require(
        admission.status == AdmissionStatus::Supported,
        format!("active backend is not admitted: {admission:?}"),
    )
}

pub fn validate_affine_descriptor(
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
    validate_affine_storage(descriptor, reference)?;
    let admission = descriptor.admission(active_target());
    require(
        admission.status != AdmissionStatus::Unsupported,
        format!("active backend rejected affine storage: {admission:?}"),
    )
}

fn validate_tokenizer(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let actual = descriptor.tokenizer_validation();
    require(
        actual.vocabulary_entries == reference.tokenizer.vocabulary_entries
            && actual.max_token_id == reference.tokenizer.max_token_id
            && actual.added_tokens == reference.tokenizer.added_tokens
            && actual.stop_token_ids == reference.tokenizer.stop_token_ids,
        format!(
            "tokenizer validation differs from reference: actual={actual:?}, \
             expected={{vocabulary_entries: {}, max_token_id: {}, added_tokens: {}, \
             stop_token_ids: {:?}}}",
            reference.tokenizer.vocabulary_entries,
            reference.tokenizer.max_token_id,
            reference.tokenizer.added_tokens,
            reference.tokenizer.stop_token_ids,
        ),
    )
}

fn validate_dense_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let mut actual = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        let WeightEncoding::Dense { dtype } = encoding else {
            return Err(validation_error("MF-100 fixture contains non-dense checkpoint storage"));
        };
        actual.insert(dtype);
    }
    let expected = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(
        actual == expected,
        format!("dtypes differ: actual={actual:?}, expected={expected:?}"),
    )
}

fn validate_affine_storage(descriptor: &ModelDescriptor, reference: &Reference) -> TestResult<()> {
    let expected = reference
        .affine
        .as_ref()
        .ok_or_else(|| validation_error("affine checkpoint reference has no affine contract"))?;
    let mut dense = BTreeSet::new();
    let mut affine = BTreeSet::new();
    for encoding in descriptor.checkpoint_encoding().weights {
        match encoding {
            WeightEncoding::Dense { dtype } => {
                dense.insert(dtype);
            },
            WeightEncoding::Affine { format } => {
                require(native_mlx(format), "checkpoint affine contract is not native MLX U32")?;
                affine.insert((
                    format.bits.get(),
                    format.group_size,
                    format.scale_dtype.as_str().to_owned(),
                ));
            },
            _ => return Err(validation_error("MF-110 fixture contains another packed encoding")),
        }
    }
    let expected_dense = reference.dtypes.iter().cloned().collect::<BTreeSet<_>>();
    require(dense == expected_dense, "dense companion dtypes differ from reference")?;
    let expected_affine = expected
        .bits
        .iter()
        .flat_map(|bits| {
            expected
                .group_sizes
                .iter()
                .map(|group| (*bits, *group, expected.parameter_dtype.clone()))
        })
        .collect::<BTreeSet<_>>();
    require(
        affine == expected_affine,
        format!("affine formats differ: actual={affine:?}, expected={expected_affine:?}"),
    )
}

fn native_mlx(format: GroupedAffineQuantization) -> bool {
    format.is_mlx_layout()
        && format.has_additive_bias()
        && format.storage_dtype == AffineStorageDType::U32
        && format.bias_dtype == Some(format.scale_dtype)
}

fn family(descriptor: &ModelDescriptor) -> Family {
    let capabilities = &descriptor.architecture_requirements().capabilities;
    if capabilities.contains(&ArchitectureCapability::DenseAndRouted) {
        Family::DenseAndRouted
    } else if capabilities.contains(&ArchitectureCapability::SharedExpert) {
        Family::SharedRouted
    } else if capabilities.contains(&ArchitectureCapability::ClampedSwiGlu)
        && capabilities.contains(&ArchitectureCapability::RoutedExperts)
    {
        Family::ClampedRouted
    } else {
        Family::Dense
    }
}
