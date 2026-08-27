mod chunked;
mod fused;
mod transform;

use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::*;
use crate::{
    CudaConfig,
    kernels::{GatedDeltaLaunch, GatedDeltaRecurrenceMode},
};

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
fn value_tiled_recurrence_matches_serial_for_partial_tile() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = GatedDeltaSpec {
        tokens: 3,
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 5,
    };
    let query = copy(&backend, &pattern(spec.tokens * spec.key_dim, 0.01))?;
    let key = copy(&backend, &pattern(spec.tokens * spec.key_dim, -0.008))?;
    let value = copy(&backend, &pattern(spec.tokens * spec.value_dim, 0.02))?;
    let alpha = copy(&backend, &pattern(spec.tokens, 0.03))?;
    let beta = copy(&backend, &pattern(spec.tokens, -0.04))?;
    let parameter = copy(&backend, &bf16s(&[0.1]))?;
    let operation = GatedDeltaRecurrence::compile(&backend.inner.compiler, spec)?;
    let mut serial = state_for(&backend, spec)?;
    let mut tiled = state_for(&backend, spec)?;
    let serial_output = execute_mode(
        &backend,
        &operation,
        &mut serial,
        (&query, &key, &value, &alpha, &beta, &parameter),
        GatedDeltaRecurrenceMode::Serial,
    )?;
    let tiled_output = execute_mode(
        &backend,
        &operation,
        &mut tiled,
        (&query, &key, &value, &alpha, &beta, &parameter),
        GatedDeltaRecurrenceMode::ValueTiled2,
    )?;
    assert_eq!(serial_output, tiled_output);
    assert_eq!(read(&backend, &serial.state)?, read(&backend, &tiled.state)?);
    Ok(())
}

type Inputs<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
);

fn execute_mode(
    backend: &CudaBackend,
    operation: &GatedDeltaRecurrence,
    state: &mut CudaGatedDeltaState,
    (query, key, value, alpha, beta, parameter): Inputs<'_>,
    mode: GatedDeltaRecurrenceMode,
) -> Result<Vec<bf16>> {
    if state.decay.len() != alpha.len() {
        state.decay = backend.inner.pool.allocate(&backend.inner.stream, alpha.len())?;
        state.update = backend.inner.pool.allocate(&backend.inner.stream, alpha.len())?;
    }
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, value.len())?;
    operation.execute_with(
        &backend.inner.stream,
        &mut GatedDeltaLaunch {
            query,
            key,
            value,
            alpha,
            beta,
            a_log: parameter,
            dt_bias: parameter,
            decay: &mut state.decay,
            update: &mut state.update,
            state: &mut state.state,
            output: &mut output,
        },
        mode,
    )?;
    read(backend, &output)
}

fn state_for(backend: &CudaBackend, spec: GatedDeltaSpec) -> Result<CudaGatedDeltaState> {
    backend.prepare_gated_delta_state(GatedDeltaStateConfig {
        key_heads: spec.key_heads,
        value_heads: spec.value_heads,
        key_dim: spec.key_dim,
        value_dim: spec.value_dim,
        convolution_kernel_size: 2,
    })
}

fn pattern(elements: usize, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let value = u8::try_from(index % 13).unwrap_or_default();
            bf16::from_f32(f32::from(value) * scale)
        })
        .collect()
}

#[test]
fn restores_recurrent_and_convolution_checkpoint_on_cuda() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut state = state(&backend)?;
    let query_key = copy(&backend, &bf16s(&[1.0; 32]))?;
    let value = copy(&backend, &bf16s(&[2.0]))?;
    let gate = copy(&backend, &bf16s(&[0.0]))?;
    let parameter = copy(&backend, &bf16s(&[0.0]))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 1)?;
    let inputs = || GatedDeltaInputs {
        query: &query_key,
        key: &query_key,
        value: &value,
        alpha: &gate,
        beta: &gate,
        a_log: &parameter,
        dt_bias: &parameter,
    };
    state.execute(1, inputs(), &mut output)?;
    let checkpoint = state.checkpoint()?;
    state.execute(1, inputs(), &mut output)?;
    let expected = read(&backend, &output)?;
    state.restore(&checkpoint)?;
    assert_eq!(state.offset(), 1);
    state.execute(1, inputs(), &mut output)?;
    assert_eq!(read(&backend, &output)?, expected);
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
