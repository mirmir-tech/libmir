use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::super::super::CudaBackend;
use crate::{
    CudaConfig, Result,
    kernels::{Mrope, MropeSpec, Rope, RopeSpec},
};

#[test]
fn mrope_matches_standard_rope_for_equal_axes() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let input = copy(&backend, &bf16s(&values))?;
    let positions = copy(&backend, &[0_u32, 1, 0, 1, 0, 1])?;
    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, 16)?;
    let mut expected = backend.inner.pool.allocate(&backend.inner.stream, 16)?;
    Mrope::compile(
        &backend.inner.compiler,
        MropeSpec {
            tokens: 2,
            heads: 1,
            head_dim: 8,
            rotary_dim: 6,
            sections: [1, 1, 1],
            interleaved: true,
            theta: 10_000.0,
        },
    )?
    .execute(&backend.inner.stream, &input, &positions, &mut actual)?;
    Rope::compile(
        &backend.inner.compiler,
        RopeSpec {
            tokens: 2,
            heads: 1,
            head_dim: 8,
            rotary_dim: 6,
            pairing_dim: 6,
            theta: 10_000.0,
        },
    )?
    .execute(&backend.inner.stream, &input, &mut expected, 0)?;
    for (actual, expected) in read(&backend, &actual)?.iter().zip(read(&backend, &expected)?) {
        assert!((actual.to_f32() - expected.to_f32()).abs() < 0.02);
    }
    Ok(())
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
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
