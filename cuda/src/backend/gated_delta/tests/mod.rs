mod transform;

use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::CudaConfig;

#[test]
fn retains_gated_delta_recurrence_on_cuda() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut state = state(&backend)?;
    let mut query_key = vec![0.0; 64];
    query_key[0] = 1.0;
    query_key[32] = 1.0;
    let query = copy(&backend, &bf16s(&query_key))?;
    let key = copy(&backend, &bf16s(&query_key))?;
    let value = copy(&backend, &bf16s(&[2.0, 4.0]))?;
    let gates = copy(&backend, &bf16s(&[0.0, 0.0]))?;
    let parameters = copy(&backend, &bf16s(&[0.0]))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 2)?;
    state.execute(
        2,
        GatedDeltaInputs {
            query: &query,
            key: &key,
            value: &value,
            alpha: &gates,
            beta: &gates,
            a_log: &parameters,
            dt_bias: &parameters,
        },
        &mut output,
    )?;
    let actual = read(&backend, &output)?;
    assert!((actual[0].to_f32() - 1.0).abs() < 0.01);
    assert!((actual[1].to_f32() - 2.25).abs() < 0.01);
    assert_eq!(state.offset(), 2);
    Ok(())
}

#[test]
fn retains_depthwise_convolution_history_on_cuda() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut state = state(&backend)?;
    let channels = channels(state.config)?;
    let weight = copy(&backend, &bf16s(&vec![1.0; channels * 2]))?;
    let mut first_values = vec![0.0; channels * 2];
    first_values[0] = 1.0;
    first_values[channels] = 2.0;
    let first = copy(&backend, &bf16s(&first_values))?;
    let mut first_output = backend.inner.pool.allocate(&backend.inner.stream, channels * 2)?;
    state.convolve_silu(2, &first, &weight, &mut first_output)?;
    let mut next_values = vec![0.0; channels];
    next_values[0] = 3.0;
    let next = copy(&backend, &bf16s(&next_values))?;
    let mut next_output = backend.inner.pool.allocate(&backend.inner.stream, channels)?;
    state.convolve_silu(1, &next, &weight, &mut next_output)?;
    let actual = read(&backend, &next_output)?[0].to_f32();
    let expected = 5.0 / (1.0 + (-5.0_f32).exp());
    assert!((actual - expected).abs() < 0.03);
    Ok(())
}

fn state(backend: &CudaBackend) -> Result<CudaGatedDeltaState> {
    backend.prepare_gated_delta_state(GatedDeltaStateConfig {
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 1,
        convolution_kernel_size: 2,
    })
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
