use std::fs;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use uuid::Uuid;

use super::*;
use crate::{CudaBackend, CudaConfig, CudaTensor, CudaTuningConfig, CudaTuningMode, Error, Result};

const TOKENS: usize = 64;
const INPUT: usize = 1_024;
const OUTPUT: usize = 3_072;

#[test]
fn autotunes_and_persists_complete_mxfp8_prefill() -> Result<()> {
    let directory =
        std::env::temp_dir().join(format!("libmir-cuda-mxfp8-tuning-{}", Uuid::new_v4()));
    let backend = CudaBackend::new(CudaConfig {
        tuning: CudaTuningConfig {
            mode: CudaTuningMode::Startup,
            cache_directory: Some(directory.clone()),
            ..CudaTuningConfig::default()
        },
        ..CudaConfig::default()
    })?;
    let weight = MxFp8CheckpointWeight {
        weight: CudaTensor::from_u32(
            "weight".into(),
            vec![OUTPUT, INPUT / 4],
            copy(&backend, &vec![0x3838_3838_u32; OUTPUT * INPUT / 4])?,
        ),
        scales: CudaTensor::from_u8(
            "scales".into(),
            vec![OUTPUT, INPUT / 32],
            copy(&backend, &vec![127_u8; OUTPUT * INPUT / 32])?,
        ),
        bias: None,
        input_features: INPUT,
        output_features: OUTPUT,
        layout: BlockProjectionLayout::Matrix,
        swizzled_scales: Arc::new(OnceLock::new()),
    };
    let operation = weight.prepare(&backend, TOKENS)?;
    let input = copy(&backend, &vec![bf16::ONE; TOKENS * INPUT])?;
    let mut output = backend.pool().allocate_zeroed(backend.stream(), TOKENS * OUTPUT)?;
    operation.execute(&input, &weight, &mut output)?;
    let actual = read(&backend, &output)?;
    assert!(actual.iter().all(|value| *value == bf16::from_f32(1_024.0)));

    let profile = fs::read_dir(&directory)?
        .next()
        .ok_or(Error::InvalidExecutionPlan("missing MXFP8 tuning profile"))??
        .path();
    let payload = fs::read_to_string(profile)?;
    assert!(payload.contains("\"quantized\""));
    assert!(payload.contains("MxFp8"));
    assert!(payload.contains("TensorCore"));
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
