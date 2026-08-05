use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{
    CudaConfig,
    kernels::{ElementwiseBf16, ShiftedRmsNorm},
};

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

#[test]
fn fused_residual_rms_norm_matches_separate_operations() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let operation = ShiftedRmsNorm::compile(&backend.inner.compiler, 2, 4, 1.0e-6, 1.0)?;
    let add = ElementwiseBf16::compile(&backend.inner.compiler, 8)?;
    let input = copy(&backend, &[1.0, -2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0].map(bf16::from_f32))?;
    let update = copy(&backend, &[0.5, 1.0, -1.5, 2.0, -2.5, 3.0, 3.5, -4.0].map(bf16::from_f32))?;
    let weight = copy(&backend, &[0.0, 0.25, -0.5, 1.0].map(bf16::from_f32))?;
    let allocate = || backend.inner.pool.allocate(&backend.inner.stream, 8);
    let mut separate_residual = allocate()?;
    let mut separate_output = allocate()?;
    add.add(&backend.inner.stream, &input, &update, &mut separate_residual)?;
    operation.execute(&backend.inner.stream, &separate_residual, &weight, &mut separate_output)?;
    let mut fused_residual = allocate()?;
    let mut fused_output = allocate()?;
    operation.execute_residual(
        &backend.inner.stream,
        &input,
        &update,
        &weight,
        &mut fused_residual,
        &mut fused_output,
    )?;
    assert_eq!(read(&backend, &fused_residual)?, read(&backend, &separate_residual)?);
    assert_eq!(read(&backend, &fused_output)?, read(&backend, &separate_output)?);
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
