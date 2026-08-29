use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{
    CudaConfig,
    kernels::{ElementwiseBf16, NvFp4Preparation, ShiftedRmsNorm, scale_elements},
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

#[test]
fn fused_residual_norm_quantization_matches_separate_kernels() -> Result<()> {
    const ROWS: usize = 3;
    const COLUMNS: usize = 128;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let norm = ShiftedRmsNorm::compile(&backend.inner.compiler, ROWS, COLUMNS, 1.0e-6, 1.0)?;
    let quantize = NvFp4Preparation::compile(&backend.inner.compiler)?;
    let values = (0..ROWS * COLUMNS)
        .map(|index| Ok(bf16::from_f32((f32::from(u16::try_from(index)?) - 173.0) / 37.0)))
        .collect::<Result<Vec<_>>>()?;
    let updates = (0..ROWS * COLUMNS)
        .map(|index| Ok(bf16::from_f32((f32::from(u16::try_from(index * 13 % 47)?) - 23.0) / 19.0)))
        .collect::<Result<Vec<_>>>()?;
    let weights = (0..COLUMNS)
        .map(|index| Ok(bf16::from_f32((f32::from(u16::try_from(index)?) - 61.0) / 257.0)))
        .collect::<Result<Vec<_>>>()?;
    let input = copy(&backend, &values)?;
    let update = copy(&backend, &updates)?;
    let weight = copy(&backend, &weights)?;
    let global_scale = copy(&backend, &[0.75_f32])?;
    let elements = ROWS * COLUMNS;
    let scales = scale_elements(ROWS, COLUMNS)?;
    let allocate_bf16 = || backend.inner.pool.allocate(&backend.inner.stream, elements);
    let mut reference_residual = allocate_bf16()?;
    let mut reference_output = allocate_bf16()?;
    norm.execute_residual(
        &backend.inner.stream,
        &input,
        &update,
        &weight,
        &mut reference_residual,
        &mut reference_output,
    )?;
    let mut reference_packed = backend.inner.pool.allocate(&backend.inner.stream, elements / 2)?;
    let mut reference_scales = backend.inner.pool.allocate_zeroed(&backend.inner.stream, scales)?;
    quantize.quantize(
        &backend.inner.stream,
        ROWS,
        COLUMNS,
        &reference_output,
        &global_scale,
        &mut reference_packed,
        &mut reference_scales,
    )?;
    let mut fused_residual = allocate_bf16()?;
    let mut fused_output = allocate_bf16()?;
    let mut fused_packed = backend.inner.pool.allocate(&backend.inner.stream, elements / 2)?;
    let mut fused_scales = backend.inner.pool.allocate_zeroed(&backend.inner.stream, scales)?;
    norm.execute_residual_nvfp4(
        &backend.inner.stream,
        &input,
        &update,
        &weight,
        &global_scale,
        &mut fused_residual,
        &mut fused_output,
        &mut fused_packed,
        &mut fused_scales,
    )?;
    assert_eq!(read(&backend, &fused_residual)?, read(&backend, &reference_residual)?);
    assert_eq!(read(&backend, &fused_output)?, read(&backend, &reference_output)?);
    assert_eq!(read(&backend, &fused_packed)?, read(&backend, &reference_packed)?);
    assert_eq!(read(&backend, &fused_scales)?, read(&backend, &reference_scales)?);
    Ok(())
}

#[test]
fn fused_bucketed_output_matches_reduce_and_residual_adds() -> Result<()> {
    const TOKENS: usize = 2;
    const ROWS: usize = 2;
    const COLUMNS: usize = 64;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let operation = ElementwiseBf16::compile(&backend.inner.compiler, COLUMNS)?;
    let input = (0..TOKENS * ROWS * COLUMNS)
        .map(|index| Ok(bf16::from_f32((f32::from(u16::try_from(index % 29)?) - 14.0) / 9.0)))
        .collect::<Result<Vec<_>>>()?;
    let input = copy(&backend, &input)?;
    let weights = copy(&backend, &[0.25, 0.75, 0.6, 0.4].map(bf16::from_f32))?;
    let positions = copy(&backend, &[2_u32, 0, 3, 1])?;
    let residual = copy(
        &backend,
        &(0..TOKENS * COLUMNS)
            .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index % 17)?) / 11.0)))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let shared = copy(
        &backend,
        &(0..TOKENS * COLUMNS)
            .map(|index| Ok(bf16::from_f32(-f32::from(u16::try_from(index % 13)?) / 7.0)))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let allocate = || backend.inner.pool.allocate(&backend.inner.stream, TOKENS * COLUMNS);
    let mut routed = allocate()?;
    let mut moe = allocate()?;
    let mut reference = allocate()?;
    operation.weighted_reduce_bucketed(
        &backend.inner.stream,
        &input,
        &weights,
        &positions,
        &mut routed,
        ROWS,
        TOKENS,
    )?;
    let add = ElementwiseBf16::compile(&backend.inner.compiler, TOKENS * COLUMNS)?;
    add.add(&backend.inner.stream, &routed, &shared, &mut moe)?;
    add.add(&backend.inner.stream, &residual, &moe, &mut reference)?;
    let mut fused = allocate()?;
    operation.weighted_reduce_bucketed_residual_shared(
        &backend.inner.stream,
        &input,
        &weights,
        &positions,
        &residual,
        &shared,
        &mut fused,
        ROWS,
        TOKENS,
    )?;
    assert_eq!(read(&backend, &fused)?, read(&backend, &reference)?);
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
