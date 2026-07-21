use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{CudaBackend, CudaConfig};

#[test]
fn mlx_split_mxfp4_experts_match_reference() -> Result<()> {
    const WIDTH: usize = 32;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let kernels = ClampedRoutedKernels::compile(
        &backend.inner.compiler,
        ClampedRoutedSpec {
            tokens: 1,
            hidden: WIDTH,
            intermediate: WIDTH,
            query_heads: 1,
            kv_heads: 1,
            head_dim: WIDTH,
            top_k: 1,
            theta: 150_000.0,
            factor: 32.0,
            initial_context: 4096.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            swiglu_limit: 7.0,
        },
    )?;
    let input = upload(&backend, &vec![bf16::from_f32(1.0); WIDTH])?;
    let blocks = upload(&backend, &vec![0x1111_1111_u32; WIDTH * (WIDTH / 8)])?;
    let scales = upload(&backend, &vec![127_u8; WIDTH * (WIDTH / 32)])?;
    let bias = upload(&backend, &vec![bf16::from_f32(0.0); WIDTH])?;
    let selected = upload(&backend, &[0_u32])?;
    let routing = upload(&backend, &[bf16::from_f32(1.0)])?;
    let mut activated = backend.inner.pool.allocate(&backend.inner.stream, WIDTH)?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, WIDTH)?;

    kernels.gate_up_mlx(
        &backend.inner.stream,
        &input,
        &blocks,
        &scales,
        &bias,
        &blocks,
        &scales,
        &bias,
        &selected,
        &mut activated,
    )?;
    kernels.down_mlx(
        &backend.inner.stream,
        &activated,
        &blocks,
        &scales,
        &bias,
        &selected,
        &routing,
        &mut output,
    )?;
    let output = read(&backend, &output)?;
    let activation = 7.0 / (1.0 + (-1.702_f32 * 7.0).exp()) * 8.0;
    let expected = activation * 16.0;
    assert!(output.iter().all(|value| (value.to_f32() - expected).abs() < 0.2));
    Ok(())
}

fn upload<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
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
