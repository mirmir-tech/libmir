use super::*;

#[test]
fn batched_convolution_matches_independent_rows() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = GatedDeltaStateConfig {
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 2,
        convolution_kernel_size: 2,
    };
    let channels = channels(config)?;
    let mut first = backend.prepare_gated_delta_state(config)?;
    let mut second = backend.prepare_gated_delta_state(config)?;
    let mut first_reference = backend.prepare_gated_delta_state(config)?;
    let mut second_reference = backend.prepare_gated_delta_state(config)?;
    let weights = copy(&backend, &pattern(channels * 2, 0.01))?;
    prime_history(&backend, &mut first, &weights, 0.02)?;
    prime_history(&backend, &mut second, &weights, -0.03)?;
    prime_history(&backend, &mut first_reference, &weights, 0.02)?;
    prime_history(&backend, &mut second_reference, &weights, -0.03)?;

    let input = copy(&backend, &pattern(channels * 2, 0.04))?;
    let mut batch_output = backend.inner.pool.allocate(&backend.inner.stream, channels * 2)?;
    let mut batch = CudaGatedDeltaBatchState::new(&backend, config, 2, 1)?;
    let states = [&mut first, &mut second];
    batch.pack(&states)?;
    batch.convolve(&input, &weights, &mut batch_output)?;

    let first_input = slice(&backend, &input, 0, channels)?;
    let second_input = slice(&backend, &input, channels, channels)?;
    let mut first_output = backend.inner.pool.allocate(&backend.inner.stream, channels)?;
    let mut second_output = backend.inner.pool.allocate(&backend.inner.stream, channels)?;
    first_reference.convolve_silu(1, &first_input, &weights, &mut first_output)?;
    second_reference.convolve_silu(1, &second_input, &weights, &mut second_output)?;
    let expected = [read(&backend, &first_output)?, read(&backend, &second_output)?].concat();
    assert_eq!(read(&backend, &batch_output)?, expected);
    Ok(())
}

#[test]
fn batched_recurrence_matches_independent_rows() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = GatedDeltaStateConfig {
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 2,
        convolution_kernel_size: 2,
    };
    let mut first = backend.prepare_gated_delta_state(config)?;
    let mut second = backend.prepare_gated_delta_state(config)?;
    let mut first_reference = backend.prepare_gated_delta_state(config)?;
    let mut second_reference = backend.prepare_gated_delta_state(config)?;
    prime(&backend, &mut first, 0.01)?;
    prime(&backend, &mut second, -0.02)?;
    prime(&backend, &mut first_reference, 0.01)?;
    prime(&backend, &mut second_reference, -0.02)?;

    let query = copy(&backend, &pattern(64, 0.015))?;
    let key = copy(&backend, &pattern(64, -0.012))?;
    let value = copy(&backend, &pattern(4, 0.08))?;
    let alpha = copy(&backend, &pattern(2, -0.03))?;
    let beta = copy(&backend, &pattern(2, 0.04))?;
    let parameter = copy(&backend, &bf16s(&[0.1]))?;
    let mut batch_output = backend.inner.pool.allocate(&backend.inner.stream, 4)?;
    let mut batch = CudaGatedDeltaBatchState::new(&backend, config, 2, 1)?;
    let mut states = [&mut first, &mut second];
    batch.pack(&states)?;
    batch.recur(
        GatedDeltaInputs {
            query: &query,
            key: &key,
            value: &value,
            alpha: &alpha,
            beta: &beta,
            a_log: &parameter,
            dt_bias: &parameter,
        },
        &mut batch_output,
    )?;

    let first_output = independent(
        &backend,
        &mut first_reference,
        (&query, &key, &value, &alpha, &beta, &parameter),
        0,
    )?;
    let second_output = independent(
        &backend,
        &mut second_reference,
        (&query, &key, &value, &alpha, &beta, &parameter),
        1,
    )?;
    let expected = [first_output, second_output].concat();
    assert_eq!(read(&backend, &batch_output)?, expected);
    batch.commit(&mut states)?;
    Ok(())
}

fn prime_history(
    backend: &CudaBackend,
    state: &mut CudaGatedDeltaState,
    weights: &DeviceBuffer<bf16>,
    scale: f32,
) -> Result<()> {
    let input = copy(backend, &pattern(channels(state.config)?, scale))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, input.len())?;
    state.convolve_silu(1, &input, weights, &mut output)
}

fn prime(backend: &CudaBackend, state: &mut CudaGatedDeltaState, scale: f32) -> Result<()> {
    let query = copy(backend, &pattern(32, scale))?;
    let key = copy(backend, &pattern(32, -scale))?;
    let value = copy(backend, &pattern(2, scale * 5.0))?;
    let alpha = copy(backend, &bf16s(&[scale]))?;
    let beta = copy(backend, &bf16s(&[-scale]))?;
    let parameter = copy(backend, &bf16s(&[0.1]))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 2)?;
    state.execute(
        1,
        GatedDeltaInputs {
            query: &query,
            key: &key,
            value: &value,
            alpha: &alpha,
            beta: &beta,
            a_log: &parameter,
            dt_bias: &parameter,
        },
        &mut output,
    )
}

fn independent(
    backend: &CudaBackend,
    state: &mut CudaGatedDeltaState,
    inputs: Inputs<'_>,
    row: usize,
) -> Result<Vec<bf16>> {
    let (query, key, value, alpha, beta, parameter) = inputs;
    let query = slice(backend, query, row * 32, 32)?;
    let key = slice(backend, key, row * 32, 32)?;
    let value = slice(backend, value, row * 2, 2)?;
    let alpha = slice(backend, alpha, row, 1)?;
    let beta = slice(backend, beta, row, 1)?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 2)?;
    state.execute(
        1,
        GatedDeltaInputs {
            query: &query,
            key: &key,
            value: &value,
            alpha: &alpha,
            beta: &beta,
            a_log: parameter,
            dt_bias: parameter,
        },
        &mut output,
    )?;
    read(backend, &output)
}

fn slice(
    backend: &CudaBackend,
    input: &DeviceBuffer<bf16>,
    offset: usize,
    elements: usize,
) -> Result<DeviceBuffer<bf16>> {
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, elements)?;
    backend
        .inner
        .stream
        .copy_device_range(input, offset..offset + elements, &mut output, 0)?;
    Ok(output)
}
