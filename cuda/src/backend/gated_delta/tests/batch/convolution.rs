use super::*;
use crate::kernels::{GatedDeltaBatchConvolution, GatedDeltaBatchConvolutionSpec};

#[test]
fn packed_prefill_convolution_in_place_preserves_old_history() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    for tokens in [1, 2, 3, 128] {
        check_history(&backend, tokens, 0)?;
        check_history(&backend, tokens, 64)?;
    }
    Ok(())
}

fn check_history(backend: &CudaBackend, tokens: usize, padding: usize) -> Result<()> {
    let rows = 5;
    let channels = 10_240;
    let kernel_size = 4;
    let spec = GatedDeltaBatchConvolutionSpec { rows, tokens, channels, kernel_size };
    let operation = GatedDeltaBatchConvolution::compile(&backend.inner.compiler, spec)?;
    let stream = &backend.inner.stream;
    let stride = channels + padding;
    let offset = padding / 2;
    let input = copy(backend, &pattern(rows * tokens * stride, 0.01))?;
    let weights = copy(backend, &pattern(channels * kernel_size, 0.02))?;
    let history = copy(backend, &pattern(rows * channels * (kernel_size - 1), -0.03))?;
    let mut next = backend.inner.pool.allocate(stream, history.len())?;
    let mut expected = backend.inner.pool.allocate(stream, rows * tokens * channels)?;
    operation.execute_strided(
        stream, &input, &weights, &history, &mut next, &mut expected, stride, offset,
    )?;
    let expected_output = read(backend, &expected)?;
    let expected_history = read(backend, &next)?;
    let mut aliased = backend.inner.pool.allocate(stream, history.len())?;
    let mut actual = backend.inner.pool.allocate(stream, expected.len())?;
    for attempt in 0..20 {
        stream.copy_device_range(&history, 0..history.len(), &mut aliased, 0)?;
        operation.execute_in_place_strided(
            stream, &input, &weights, &mut aliased, &mut actual, stride, offset,
        )?;
        let output = read(backend, &actual)?;
        let difference = output.iter().zip(&expected_output).position(|(a, b)| a != b);
        assert!(
            difference.is_none(),
            "tokens={tokens} padding={padding} attempt={attempt} first={difference:?}"
        );
        assert_eq!(read(backend, &aliased)?, expected_history);
    }
    Ok(())
}
