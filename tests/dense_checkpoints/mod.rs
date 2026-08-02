#![cfg(any(feature = "cuda", feature = "metal"))]

mod affine;
mod bitsandbytes;
mod fixture;
mod logits;
mod modelopt;
mod policy;
mod run;

use std::error::Error;

use fixture::{Catalog, load_reference, required_path};
use libmir::{GenerationOverrides, Library, ModelDescriptor, RuntimeConfig};
use policy::configure_cuda_policy;

type TestResult<T> = Result<T, Box<dyn Error>>;

const CATALOG: &str = include_str!("../../validation/dense-checkpoints.toml");
const REFERENCE_EXAMPLE: &str =
    include_str!("../../validation/dense-checkpoint-reference.example.toml");

#[test]
fn dense_checkpoint_catalog_covers_every_semantic_family() -> TestResult<()> {
    let catalog = Catalog::parse(CATALOG)?;
    catalog.validate()?;
    fixture::Reference::parse(REFERENCE_EXAMPLE)?.validate_for(fixture::Family::Dense)?;
    Ok(())
}

#[test]
fn dense_checkpoint_resource_gate_rejects_legacy_throughput_fields() {
    let legacy = REFERENCE_EXAMPLE.replacen(
        "max_decode_active_bytes = 8500000000",
        "max_decode_active_bytes = 8500000000\nmin_decode_tokens_per_second = 1.0",
        1,
    );
    assert!(fixture::Reference::parse(&legacy).is_err());
}

#[test]
#[ignore = "V2-V4 gate; configure all model and reference variables from dense-checkpoints.toml"]
fn validates_dense_checkpoint_matrix_v2_to_v4() -> TestResult<()> {
    let catalog = Catalog::parse(CATALOG)?;
    catalog.validate()?;
    let selected = std::env::var("MIRMIR_DENSE_FIXTURE_FAMILY").ok();
    let mut matched = false;
    for fixture in catalog.fixtures {
        if selected.as_deref().is_some_and(|family| family != fixture.family.as_str()) {
            continue;
        }
        matched = true;
        let model_path = required_path(&fixture.model_env)?;
        let reference_path = required_path(&fixture.reference_env)?;
        let reference = load_reference(&reference_path)?;
        reference.validate_for(fixture.family)?;

        let mut config = RuntimeConfig::default();
        config.kv_cache.block_count = reference.kv_cache_blocks;
        config.automatic_kv_cache = false;
        config.scheduler.max_batch_requests = 2;
        config.scheduler.decode_batch_wait_us = 50_000;
        configure_cuda_policy(&mut config)?;
        let descriptor = ModelDescriptor::inspect(&model_path, GenerationOverrides::default())?;
        let library = Library::new(config);
        let baseline = library.memory_snapshot()?;
        fixture::validate_descriptor(&descriptor, fixture.family, &reference)?;
        let model = library.load(&model_path, GenerationOverrides::default(), &mut |_event| {})?;
        let loaded = library.memory_snapshot()?;
        run::validate(&library, &model, &reference, &baseline, &loaded)?;
        model.unload()?;
    }
    fixture::require(
        matched,
        format!(
            "MIRMIR_DENSE_FIXTURE_FAMILY does not name a catalog family: {}",
            selected.as_deref().unwrap_or_default()
        ),
    )?;
    Ok(())
}

#[test]
#[ignore = "MF-110 V2-V4 gate; set MIRMIR_AFFINE_MODEL and MIRMIR_AFFINE_REFERENCE"]
fn validates_affine_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_AFFINE_MODEL",
        "MIRMIR_AFFINE_REFERENCE",
        fixture::Reference::validate_affine_for,
        fixture::validate_affine_descriptor,
    )
}

#[test]
#[ignore = "MF-120 V2-V4 gate; set MIRMIR_PACKED_INT8_MODEL and MIRMIR_PACKED_INT8_REFERENCE"]
fn validates_packed_int8_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_PACKED_INT8_MODEL",
        "MIRMIR_PACKED_INT8_REFERENCE",
        fixture::Reference::validate_packed_int8_for,
        fixture::validate_packed_int8_descriptor,
    )
}

#[test]
#[ignore = "MF-120 V2-V4 gate; set MIRMIR_PACKED_INT4_MODEL and MIRMIR_PACKED_INT4_REFERENCE"]
fn validates_packed_int4_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_PACKED_INT4_MODEL",
        "MIRMIR_PACKED_INT4_REFERENCE",
        fixture::Reference::validate_packed_int4_for,
        fixture::validate_packed_int4_descriptor,
    )
}

#[test]
#[ignore = "MF-120 V2-V4 gate; set MIRMIR_AWQ_MODEL and MIRMIR_AWQ_REFERENCE"]
fn validates_awq_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_AWQ_MODEL",
        "MIRMIR_AWQ_REFERENCE",
        fixture::Reference::validate_awq_for,
        fixture::validate_awq_descriptor,
    )
}

#[test]
#[ignore = "MF-120 V2-V4 gate; set MIRMIR_GPTQ_MODEL and MIRMIR_GPTQ_REFERENCE"]
fn validates_gptq_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_GPTQ_MODEL",
        "MIRMIR_GPTQ_REFERENCE",
        fixture::Reference::validate_gptq_for,
        fixture::validate_gptq_descriptor,
    )
}

#[test]
#[ignore = "MF-130 V2-V4 gate; set MIRMIR_FP8_MODEL and MIRMIR_FP8_REFERENCE"]
fn validates_float8_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_FP8_MODEL",
        "MIRMIR_FP8_REFERENCE",
        fixture::Reference::validate_float8_for,
        fixture::validate_float8_descriptor,
    )
}

#[test]
#[ignore = "MF-130 V2-V4 gate; set MIRMIR_MXFP8_MODEL and MIRMIR_MXFP8_REFERENCE"]
fn validates_mxfp8_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint(
        "MIRMIR_MXFP8_MODEL",
        "MIRMIR_MXFP8_REFERENCE",
        fixture::Reference::validate_mxfp8_for,
        fixture::validate_mxfp8_descriptor,
    )
}

#[test]
#[ignore = "MF-130 V2-V4 gate; set MIRMIR_MXFP4_ROUTED_MODEL and MIRMIR_MXFP4_ROUTED_REFERENCE"]
fn validates_mxfp4_routed_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint_for(
        "MIRMIR_MXFP4_ROUTED_MODEL",
        "MIRMIR_MXFP4_ROUTED_REFERENCE",
        fixture::Family::ClampedRouted,
        fixture::Reference::validate_mxfp4_for,
        fixture::validate_mxfp4_descriptor,
    )
}

#[test]
#[ignore = "MF-130 V2-V4 gate; set MIRMIR_MXFP4_GATHERED_MODEL and MIRMIR_MXFP4_GATHERED_REFERENCE"]
fn validates_mxfp4_gathered_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint_for(
        "MIRMIR_MXFP4_GATHERED_MODEL",
        "MIRMIR_MXFP4_GATHERED_REFERENCE",
        fixture::Family::SharedRouted,
        fixture::Reference::validate_mxfp4_for,
        fixture::validate_mxfp4_descriptor,
    )
}

#[test]
#[ignore = "MF-130 V2-V4 gate; set MIRMIR_NVFP4_MODEL and MIRMIR_NVFP4_REFERENCE"]
fn validates_nvfp4_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint_for(
        "MIRMIR_NVFP4_MODEL",
        "MIRMIR_NVFP4_REFERENCE",
        fixture::Family::DenseAndRouted,
        fixture::Reference::validate_nvfp4_for,
        fixture::validate_nvfp4_descriptor,
    )
}

fn validate_format_checkpoint(
    model_env: &str,
    reference_env: &str,
    validate_reference: fn(&fixture::Reference, fixture::Family) -> TestResult<()>,
    validate_descriptor: fn(
        &ModelDescriptor,
        fixture::Family,
        &fixture::Reference,
    ) -> TestResult<()>,
) -> TestResult<()> {
    validate_format_checkpoint_for(
        model_env,
        reference_env,
        fixture::Family::Dense,
        validate_reference,
        validate_descriptor,
    )
}

fn validate_format_checkpoint_for(
    model_env: &str,
    reference_env: &str,
    family: fixture::Family,
    validate_reference: fn(&fixture::Reference, fixture::Family) -> TestResult<()>,
    validate_descriptor: fn(
        &ModelDescriptor,
        fixture::Family,
        &fixture::Reference,
    ) -> TestResult<()>,
) -> TestResult<()> {
    let model_path = required_path(model_env)?;
    let reference_path = required_path(reference_env)?;
    let reference = load_reference(&reference_path)?;
    validate_reference(&reference, family)?;
    let mut config = RuntimeConfig::default();
    config.memory.reserve_percent = Some(1);
    config.memory.reserve_bytes = Some(0);
    config.kv_cache.block_count = reference.kv_cache_blocks;
    config.automatic_kv_cache = false;
    config.scheduler.max_batch_requests = 2;
    config.scheduler.decode_batch_wait_us = 50_000;
    configure_cuda_policy(&mut config)?;
    let descriptor = ModelDescriptor::inspect(&model_path, GenerationOverrides::default())?;
    let library = Library::new(config);
    let baseline = library.memory_snapshot()?;
    validate_descriptor(&descriptor, family, &reference)?;
    let model = library.load(&model_path, GenerationOverrides::default(), &mut |_event| {})?;
    let loaded = library.memory_snapshot()?;
    run::validate(&library, &model, &reference, &baseline, &loaded)?;
    model.unload()?;
    Ok(())
}
