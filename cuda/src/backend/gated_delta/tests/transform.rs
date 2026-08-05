use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{
    CudaConfig, Result,
    kernels::{GatedDeltaTransformSpec, GatedDeltaTransforms},
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
