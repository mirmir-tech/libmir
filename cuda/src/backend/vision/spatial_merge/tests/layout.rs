use mircuda::{DeviceElement, bf16};

use crate::{CudaConfig, Result, backend::CudaBackend};

#[test]
fn reorders_channel_first_patches_for_channel_last_weights() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let input = copy(&backend, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 6)?;
    let operation = crate::kernels::VisionPatchLayout::compile(&backend.inner.compiler)?;
    operation.cthw_to_thwc(&backend.inner.stream, &input, &mut output, [1, 3, 2, 1])?;
    assert_eq!(read(&backend, &output)?, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    Ok(())
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<mircuda::DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read(backend: &CudaBackend, source: &mircuda::DeviceBuffer<bf16>) -> Result<Vec<f32>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?.into_iter().map(bf16::to_f32).collect())
}
