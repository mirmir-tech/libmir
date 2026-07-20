use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{CudaConfig, kernels::ShiftedRmsNorm};

#[test]
fn shifted_rms_norm_applies_the_structural_weight_offset() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let operation = ShiftedRmsNorm::compile(&backend.inner.compiler, 1, 2, 0.0, 1.0)?;
    let input = copy(&backend, &[bf16::from_f32(1.0), bf16::from_f32(2.0)])?;
    let weight = copy(&backend, &[bf16::ZERO; 2])?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 2)?;
    operation.execute(&backend.inner.stream, &input, &weight, &mut output)?;
    let actual = read(&backend, &output)?;
    let inverse = 2.5_f32.sqrt().recip();
    let second = 2.0 * inverse;
    assert!((actual[0].to_f32() - inverse).abs() < 0.005);
    assert!((actual[1].to_f32() - second).abs() < 0.005);
    Ok(())
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
