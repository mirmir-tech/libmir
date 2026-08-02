use std::fs;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use uuid::Uuid;

use super::super::*;
use crate::{CudaConfig, CudaTuningConfig, CudaTuningMode};

const PREFILL_TOKENS: usize = 64;
const INPUT: usize = 896;
const OUTPUT: usize = 4_864;

#[test]
fn profiles_and_persists_dynamic_e4m3_tensor_core() -> Result<()> {
    let (backend, directory) = backend()?;
    let weight = checkpoint_weight(
        &backend,
        Some(CudaTensor::from_f32(
            "scales".into(),
            vec![OUTPUT],
            copy(&backend, &vec![1.0_f32; OUTPUT])?,
        )),
        None,
        DirectFp8Scale::OutputChannel,
        DirectFp8Activation::DynamicE4M3Token,
    )?;
    execute_and_validate(&backend, &weight, PREFILL_TOKENS)?;
    validate_profile(&directory, "DirectFp8DynamicE4M3OutputChannel", "TensorCore")
}

#[test]
fn profiles_and_persists_static_e4m3_tensor_core() -> Result<()> {
    let (backend, directory) = backend()?;
    let weight = static_weight(&backend)?;
    execute_and_validate(&backend, &weight, PREFILL_TOKENS)?;
    validate_profile(&directory, "DirectFp8StaticE4M3", "TensorCore")
}

#[test]
fn profiles_static_e4m3_decode_independently() -> Result<()> {
    let (backend, directory) = backend()?;
    let weight = static_weight(&backend)?;
    execute_and_validate(&backend, &weight, 1)?;
    validate_profile(&directory, "DirectFp8StaticE4M3", "Portable")
}

#[test]
fn profiles_unscaled_e5m2_weight_only_tensor_core() -> Result<()> {
    let (backend, directory) = backend()?;
    let weight = e5m2_weight(&backend)?;
    execute_and_validate(&backend, &weight, PREFILL_TOKENS)?;
    validate_profile(&directory, "DirectFp8Bf16E5M2WeightOnly", "TensorCore")
}

#[test]
fn profiles_unscaled_e5m2_decode_independently() -> Result<()> {
    let (backend, directory) = backend()?;
    let weight = e5m2_weight(&backend)?;
    execute_and_validate(&backend, &weight, 1)?;
    validate_profile(&directory, "DirectFp8Bf16E5M2WeightOnly", "Portable")
}

fn backend() -> Result<(CudaBackend, std::path::PathBuf)> {
    let directory = std::env::temp_dir().join(format!("libmir-cuda-direct-fp8-{}", Uuid::new_v4()));
    let backend = CudaBackend::new(CudaConfig {
        tuning: CudaTuningConfig {
            mode: CudaTuningMode::Startup,
            cache_directory: Some(directory.clone()),
            ..CudaTuningConfig::default()
        },
        ..CudaConfig::default()
    })?;
    Ok((backend, directory))
}

fn checkpoint_weight(
    backend: &CudaBackend,
    scales: Option<CudaTensor>,
    input_scale: Option<CudaTensor>,
    scale: DirectFp8Scale,
    activation: DirectFp8Activation,
) -> Result<DirectFp8CheckpointWeight> {
    Ok(DirectFp8CheckpointWeight {
        weight: CudaTensor::from_f8_e4m3(
            "weight".into(),
            vec![OUTPUT, INPUT],
            copy(backend, &vec![0x38_u8; OUTPUT * INPUT])?,
        ),
        scales,
        input_scale,
        bias: None,
        input_features: INPUT,
        output_features: OUTPUT,
        format: DirectFp8Format::E4M3,
        scale,
        inverse_scale: false,
        activation,
    })
}

fn static_weight(backend: &CudaBackend) -> Result<DirectFp8CheckpointWeight> {
    checkpoint_weight(
        backend,
        Some(CudaTensor::from_bf16("scales".into(), Vec::new(), copy(backend, &[bf16::ONE])?)),
        Some(CudaTensor::from_bf16(
            "input_scale".into(),
            Vec::new(),
            copy(backend, &[bf16::ONE])?,
        )),
        DirectFp8Scale::Tensor,
        DirectFp8Activation::StaticE4M3Tensor,
    )
}

fn e5m2_weight(backend: &CudaBackend) -> Result<DirectFp8CheckpointWeight> {
    Ok(DirectFp8CheckpointWeight {
        weight: CudaTensor::from_f8_e5m2(
            "weight".into(),
            vec![OUTPUT, INPUT],
            copy(backend, &vec![0x3c_u8; OUTPUT * INPUT])?,
        ),
        scales: None,
        input_scale: None,
        bias: None,
        input_features: INPUT,
        output_features: OUTPUT,
        format: DirectFp8Format::E5M2,
        scale: DirectFp8Scale::Tensor,
        inverse_scale: false,
        activation: DirectFp8Activation::Bf16,
    })
}

fn execute_and_validate(
    backend: &CudaBackend,
    weight: &DirectFp8CheckpointWeight,
    tokens: usize,
) -> Result<()> {
    let operation = weight.prepare(backend, tokens)?;
    let input = copy(backend, &vec![bf16::ONE; tokens * INPUT])?;
    let mut output = backend.pool().allocate_zeroed(backend.stream(), tokens * OUTPUT)?;
    operation.execute(&input, weight, &mut output)?;
    assert!(read(backend, &output)?.iter().all(|value| *value == bf16::from_f32(896.0)));
    Ok(())
}

fn validate_profile(directory: &std::path::Path, format: &str, execution: &str) -> Result<()> {
    let profile = fs::read_dir(directory)?
        .next()
        .ok_or(Error::InvalidExecutionPlan("missing direct FP8 profile"))??
        .path();
    let payload = fs::read_to_string(profile)?;
    assert!(payload.contains(format));
    assert!(payload.contains(execution));
    fs::remove_dir_all(directory)?;
    Ok(())
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read(backend: &CudaBackend, values: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    backend.stream().copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
