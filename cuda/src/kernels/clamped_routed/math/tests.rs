use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{CudaBackend, CudaConfig};

#[test]
fn mlx_split_mxfp4_experts_match_reference() -> Result<()> {
    const WIDTH: usize = 32;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (_, stream, pool, compiler) = backend.test_resources();
    let kernels = ClampedRoutedKernels::compile(
        compiler,
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
    let input = upload(&backend, &[bf16::from_f32(1.0); WIDTH])?;
    let blocks = upload(&backend, &[0x1111_1111_u32; WIDTH * (WIDTH / 8)])?;
    let scales = upload(&backend, &[127_u8; WIDTH * (WIDTH / 32)])?;
    let bias = upload(&backend, &[bf16::from_f32(0.0); WIDTH])?;
    let selected = upload(&backend, &[0_u32])?;
    let routing = upload(&backend, &[bf16::from_f32(1.0)])?;
    let mut activated = pool.allocate(stream, WIDTH)?;
    let mut output = pool.allocate(stream, WIDTH)?;

    kernels.gate_up_mlx(
        stream, &input, &blocks, &scales, &bias, &blocks, &scales, &bias, &selected, &mut activated,
    )?;
    kernels
        .down_mlx(stream, &activated, &blocks, &scales, &bias, &selected, &routing, &mut output)?;
    let output = read(&backend, &output)?;
    let activation = 7.0 / (1.0 + (-1.702_f32 * 7.0).exp()) * 8.0;
    let expected = activation * 16.0;
    assert!(output.iter().all(|value| (value.to_f32() - expected).abs() < 0.2));
    Ok(())
}

#[test]
fn route_parallel_down_matches_fused_reduction_for_both_layouts() -> Result<()> {
    const WIDTH: usize = 32;
    const TOKENS: usize = 2;
    const TOP_K: usize = 2;
    const ROUTES: usize = TOKENS * TOP_K;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (_, stream, pool, compiler) = backend.test_resources();
    let kernels = ClampedRoutedKernels::compile(
        compiler,
        ClampedRoutedSpec {
            tokens: TOKENS,
            hidden: WIDTH,
            intermediate: WIDTH,
            query_heads: 1,
            kv_heads: 1,
            head_dim: WIDTH,
            top_k: TOP_K,
            theta: 150_000.0,
            factor: 32.0,
            initial_context: 4096.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            swiglu_limit: 7.0,
        },
    )?;
    let activated = upload(&backend, &[bf16::from_f32(0.5); ROUTES * WIDTH])?;
    let mlx_blocks = upload(&backend, &[0x1111_1111_u32; WIDTH * (WIDTH / 8)])?;
    let native_blocks = upload(&backend, &[0x11_u8; WIDTH * (WIDTH / 32) * 16])?;
    let scales = upload(&backend, &[127_u8; WIDTH * (WIDTH / 32)])?;
    let bias = upload(&backend, &[bf16::from_f32(0.25); WIDTH])?;
    let selected = upload(&backend, &[0_u32; ROUTES])?;
    let routing = upload(
        &backend,
        &[bf16::from_f32(0.25), bf16::from_f32(0.75), bf16::from_f32(0.6), bf16::from_f32(0.4)],
    )?;

    let mut fused = pool.allocate(stream, TOKENS * WIDTH)?;
    let mut parallel = pool.allocate(stream, TOKENS * WIDTH)?;
    let mut partial = pool.allocate(stream, ROUTES * WIDTH)?;
    kernels.down_mlx(
        stream, &activated, &mlx_blocks, &scales, &bias, &selected, &routing, &mut fused,
    )?;
    kernels.down_routes_mlx(
        stream, &activated, &mlx_blocks, &scales, &bias, &selected, &routing, &mut partial,
        &mut parallel,
    )?;
    assert_eq!(read(&backend, &fused)?, read(&backend, &parallel)?);

    kernels.down_native(
        stream, &activated, &native_blocks, &scales, &bias, &selected, &routing, &mut fused,
    )?;
    kernels.down_routes_native(
        stream, &activated, &native_blocks, &scales, &bias, &selected, &routing, &mut partial,
        &mut parallel,
    )?;
    assert_eq!(read(&backend, &fused)?, read(&backend, &parallel)?);
    Ok(())
}

fn upload<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let (context, stream, pool, _) = backend.test_resources();
    let mut host = context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let (context, stream, _, _) = backend.test_resources();
    let mut host = context.allocate_pinned(source.len())?;
    stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
