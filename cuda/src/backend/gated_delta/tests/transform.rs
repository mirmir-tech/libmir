use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{
    CudaConfig, Result,
    kernels::{
        DirectFp8Activation, DirectFp8NormGate, DirectFp8Scale, DirectFp8Spec,
        GatedDeltaTransformSpec, GatedDeltaTransforms,
    },
};

#[test]
fn transforms_gated_delta_projections_on_cuda() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let transforms = GatedDeltaTransforms::compile(
        &backend.inner.compiler,
        GatedDeltaTransformSpec {
            tokens: 1,
            key_heads: 1,
            value_heads: 1,
            key_dim: 32,
            value_dim: 4,
            epsilon: 1.0e-6,
            norm_weight_shift: 1.0,
        },
    )?;
    let mut mixed_values = vec![2.0; 32];
    mixed_values.extend(vec![3.0; 32]);
    mixed_values.extend([1.0, 2.0, 3.0, 4.0]);
    let mixed = copy(&backend, &bf16s(&mixed_values))?;
    let mut value = allocate(&backend, 4)?;
    let mut normalized_query = allocate(&backend, 32)?;
    let mut normalized_key = allocate(&backend, 32)?;
    transforms.split_normalize(
        &backend.inner.stream,
        &mixed,
        &mut normalized_query,
        &mut normalized_key,
        &mut value,
    )?;
    let gate = copy(&backend, &bf16s(&[1.0; 4]))?;
    let weight = copy(&backend, &bf16s(&[0.0; 4]))?;
    let mut gated = allocate(&backend, 4)?;
    transforms.norm_gate(&backend.inner.stream, &value, &gate, &weight, &mut gated)?;

    let actual_query = read(&backend, &normalized_query)?;
    let actual_key = read(&backend, &normalized_key)?;
    let actual_value = read(&backend, &value)?;
    let actual_gated = read(&backend, &gated)?;
    assert!((actual_query[0].to_f32() - 1.0 / 32.0).abs() < 0.001);
    assert!((actual_key[0].to_f32() - 1.0 / 32.0_f32.sqrt()).abs() < 0.002);
    assert_eq!(actual_value, bf16s(&[1.0, 2.0, 3.0, 4.0]));
    let rms = (7.5_f32 + 1.0e-6).sqrt();
    let silu = 1.0 / (1.0 + (-1.0_f32).exp());
    for (index, actual) in actual_gated.iter().enumerate() {
        let expected = f32::from(u16::try_from(index + 1)?) / rms * silu;
        assert!((actual.to_f32() - expected).abs() < 0.01);
    }
    Ok(())
}

#[test]
fn fused_norm_gate_dynamic_fp8_matches_composed_path() -> Result<()> {
    const TOKENS: usize = 2;
    const HEADS: usize = 2;
    const HEAD_WIDTH: usize = 128;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let columns = HEADS * HEAD_WIDTH;
    let elements = TOKENS * columns;
    let values = (0..elements)
        .map(|index| -> Result<f32> { Ok((f32::from(u16::try_from(index % 29)?) - 14.0) * 0.125) })
        .collect::<Result<Vec<_>>>()?;
    let activation_values = (0..elements)
        .map(|index| -> Result<f32> { Ok((f32::from(u16::try_from(index % 17)?) - 8.0) * 0.25) })
        .collect::<Result<Vec<_>>>()?;
    let input = copy(&backend, &bf16s(&values))?;
    let gate = copy(&backend, &bf16s(&activation_values))?;
    let weight = copy(&backend, &bf16s(&vec![0.0; HEAD_WIDTH]))?;
    let transforms = GatedDeltaTransforms::compile(
        &backend.inner.compiler,
        GatedDeltaTransformSpec {
            tokens: TOKENS,
            key_heads: 1,
            value_heads: HEADS,
            key_dim: HEAD_WIDTH,
            value_dim: HEAD_WIDTH,
            epsilon: 1.0e-6,
            norm_weight_shift: 1.0,
        },
    )?;
    let mut gated = allocate(&backend, elements)?;
    transforms.norm_gate(&backend.inner.stream, &input, &gate, &weight, &mut gated)?;
    let spec = DirectFp8Spec::new(
        TOKENS,
        columns,
        columns,
        DirectFp8Scale::Tensor,
        false,
        DirectFp8Activation::DynamicE4M3Token,
    )?;
    let fused = DirectFp8NormGate::compile(&backend.inner.compiler)?;
    let mut expected = backend.inner.pool.allocate(&backend.inner.stream, elements)?;
    let mut expected_scales = backend.inner.pool.allocate(&backend.inner.stream, TOKENS)?;
    fused.quantize_reference(
        &backend.inner.stream,
        spec,
        &gated,
        &mut expected,
        &mut expected_scales,
    )?;
    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, elements)?;
    let mut actual_scales = backend.inner.pool.allocate(&backend.inner.stream, TOKENS)?;
    fused.execute(
        &backend.inner.stream,
        spec,
        &input,
        &gate,
        &weight,
        &mut actual,
        &mut actual_scales,
        HEADS,
        columns,
        0,
        1.0e-6,
        1.0,
    )?;
    assert_eq!(read(&backend, &actual)?, read(&backend, &expected)?);
    assert_eq!(read(&backend, &actual_scales)?, read(&backend, &expected_scales)?);
    Ok(())
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn allocate(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    Ok(backend.inner.pool.allocate(&backend.inner.stream, elements)?)
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
