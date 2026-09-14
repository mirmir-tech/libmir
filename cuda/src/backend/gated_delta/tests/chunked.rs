use super::*;
#[test]
fn chunked_prefill_matches_serial_recurrence() -> Result<()> {
    compare(32, 67)
}

#[test]
fn chunked_48_head_prefill_matches_serial_recurrence() -> Result<()> {
    for tokens in [64, 67, 128, 257, 960, 1008, 2032, 2048] {
        compare(48, tokens)?;
    }
    Ok(())
}

fn compare(heads: usize, tokens: usize) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = GatedDeltaSpec {
        tokens,
        key_heads: 16,
        value_heads: heads,
        key_dim: 128,
        value_dim: 128,
    };
    let key_elements = spec.tokens * spec.key_heads * spec.key_dim;
    let value_elements = spec.tokens * spec.value_heads * spec.value_dim;
    let gate_elements = spec.tokens * spec.value_heads;
    let query = copy(&backend, &signal(key_elements, 17, 0.15))?;
    let key = copy(&backend, &signal(key_elements, 29, 0.15))?;
    let value = copy(&backend, &signal(value_elements, 41, 0.1))?;
    let alpha = copy(&backend, &signal(gate_elements, 53, 0.2))?;
    let beta = copy(&backend, &signal(gate_elements, 67, 0.2))?;
    let a_log = copy(&backend, &bf16s(&vec![-2.0; spec.value_heads]))?;
    let dt_bias = copy(&backend, &bf16s(&vec![0.0; spec.value_heads]))?;
    assert!(crate::kernels::GatedDeltaChunked::supports((12, 1), spec));
    let serial = GatedDeltaRecurrence::compile(&backend.inner.compiler, spec)?;
    let mut serial_state = state_for(&backend, spec)?;
    let mut chunked_state = state_for(&backend, spec)?;
    let state_elements = spec.value_heads * spec.value_dim * spec.key_dim;
    let initial_state = (0..state_elements)
        .map(|index| f32::from(u8::try_from(index % 17).unwrap_or_default()) * 0.000_01)
        .collect::<Vec<_>>();
    serial_state.state = copy(&backend, &initial_state)?;
    chunked_state.state = copy(&backend, &initial_state)?;
    resize_gates(&backend, &mut serial_state, gate_elements)?;
    resize_gates(&backend, &mut chunked_state, gate_elements)?;
    let mut serial_output = backend.inner.pool.allocate(&backend.inner.stream, value_elements)?;
    let mut chunked_output = backend.inner.pool.allocate(&backend.inner.stream, value_elements)?;
    for _ in 0..2 {
        serial.execute_with(
            &backend.inner.stream,
            &mut launch(
                &mut serial_state,
                &mut serial_output,
                (&query, &key, &value, &alpha, &beta, &a_log, &dt_bias),
            ),
            GatedDeltaRecurrenceMode::Serial,
        )?;
        backend.poison_gated_delta_workspace()?;
        chunked_state.execute(
            spec.tokens,
            GatedDeltaInputs {
                query: &query,
                key: &key,
                value: &value,
                alpha: &alpha,
                beta: &beta,
                a_log: &a_log,
                dt_bias: &dt_bias,
            },
            &mut chunked_output,
        )?;
        let output_error =
            max_bf16_error(&read(&backend, &serial_output)?, &read(&backend, &chunked_output)?);
        let state_error = max_f32_error(
            &read(&backend, &serial_state.state)?,
            &read(&backend, &chunked_state.state)?,
        );
        assert!(output_error < 0.01, "output error {output_error}");
        assert!(state_error < 0.01, "state error {state_error}");
    }
    Ok(())
}

type ChunkInputs<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
);

fn launch<'a>(
    state: &'a mut CudaGatedDeltaState,
    output: &'a mut DeviceBuffer<bf16>,
    (query, key, value, alpha, beta, a_log, dt_bias): ChunkInputs<'a>,
) -> GatedDeltaLaunch<'a> {
    GatedDeltaLaunch {
        query,
        key,
        value,
        alpha,
        beta,
        a_log,
        dt_bias,
        decay: &mut state.decay,
        update: &mut state.update,
        state: &mut state.state,
        output,
    }
}

fn resize_gates(
    backend: &CudaBackend,
    state: &mut CudaGatedDeltaState,
    elements: usize,
) -> Result<()> {
    state.decay = backend.inner.pool.allocate(&backend.inner.stream, elements)?;
    state.update = backend.inner.pool.allocate(&backend.inner.stream, elements)?;
    Ok(())
}

fn max_bf16_error(left: &[bf16], right: &[bf16]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            assert!(left.to_f32().is_finite() && right.to_f32().is_finite());
            (left.to_f32() - right.to_f32()).abs()
        })
        .fold(0.0, f32::max)
}

fn max_f32_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            assert!(left.is_finite() && right.is_finite());
            (left - right).abs()
        })
        .fold(0.0, f32::max)
}

fn signal(elements: usize, seed: u64, scale: f32) -> Vec<bf16> {
    let mut state = seed;
    (0..elements)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let sample = u16::try_from(state >> 48).unwrap_or_default();
            bf16::from_f32((f32::from(sample) / 32_768.0 - 1.0) * scale)
        })
        .collect()
}
