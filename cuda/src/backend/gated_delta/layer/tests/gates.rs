use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::super::super::*;
use crate::{
    Bf16LinearPair, CudaConfig, CudaTensor, ExecutionPhase,
    kernels::{GatedDeltaAlphaBeta, GatedDeltaAlphaBetaSplit},
};

const TOKENS: usize = 67;
const COLUMNS: usize = 128;
const HEADS: usize = 8;

#[test]
fn paired_alpha_beta_matches_direct_projection() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let input = copy(&backend, &pattern(TOKENS * COLUMNS, 0.01))?;
    let alpha_weight = copy(&backend, &pattern(HEADS * COLUMNS, 0.005))?;
    let beta_weight = copy(&backend, &pattern(HEADS * COLUMNS, -0.007))?;
    let mut direct_alpha = allocate(&backend, TOKENS * HEADS)?;
    let mut direct_beta = allocate(&backend, TOKENS * HEADS)?;
    GatedDeltaAlphaBeta::compile(&backend.inner.compiler, TOKENS, COLUMNS, HEADS)?.execute(
        &backend.inner.stream,
        &input,
        &alpha_weight,
        &beta_weight,
        &mut direct_alpha,
        &mut direct_beta,
    )?;

    let alpha = CudaTensor::from_bf16("alpha".into(), vec![HEADS, COLUMNS], alpha_weight.clone());
    let beta = CudaTensor::from_bf16("beta".into(), vec![HEADS, COLUMNS], beta_weight.clone());
    let weights = backend.pack_bf16_linear_pair(&alpha, &beta)?;
    let mut projection =
        Bf16LinearPair::new(&backend, ExecutionPhase::Prefill, TOKENS, COLUMNS, HEADS)?;
    let mut packed = allocate(&backend, TOKENS * HEADS * 2)?;
    projection.execute(&input, &weights, &mut packed)?;
    let split = GatedDeltaAlphaBetaSplit::compile(&backend.inner.compiler, TOKENS, HEADS)?;
    let mut paired_alpha = allocate(&backend, TOKENS * HEADS)?;
    let mut paired_beta = allocate(&backend, TOKENS * HEADS)?;
    split.execute(&backend.inner.stream, &packed, &mut paired_alpha, &mut paired_beta)?;

    assert!(max_error(&read(&backend, &direct_alpha)?, &read(&backend, &paired_alpha)?) < 0.01);
    assert!(max_error(&read(&backend, &direct_beta)?, &read(&backend, &paired_beta)?) < 0.01);
    Ok(())
}

fn pattern(elements: usize, scale: f32) -> Vec<bf16> {
    const VALUES: [f32; 16] =
        [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    (0..elements)
        .map(|index| bf16::from_f32(VALUES[index % VALUES.len()] * scale))
        .collect()
}

fn max_error(left: &[bf16], right: &[bf16]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0, f32::max)
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
