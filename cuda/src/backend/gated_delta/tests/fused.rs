use super::*;
use crate::kernels::{GatedDeltaTransformSpec, GatedDeltaTransforms};

#[test]
fn fused_convolution_split_matches_bf16_intermediate() -> Result<()> {
    const TOKENS: usize = 3;
    const KEY_HEADS: usize = 1;
    const VALUE_HEADS: usize = 2;
    const KEY_DIM: usize = 128;
    const VALUE_DIM: usize = 128;
    const KERNEL: usize = 2;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = GatedDeltaStateConfig {
        key_heads: KEY_HEADS,
        value_heads: VALUE_HEADS,
        key_dim: KEY_DIM,
        value_dim: VALUE_DIM,
        convolution_kernel_size: KERNEL,
    };
    let channels = channels(config)?;
    let input = copy(&backend, &pattern(TOKENS * channels, 0.015))?;
    let weight = copy(&backend, &pattern(channels * KERNEL, -0.02))?;
    let mut split_state = backend.prepare_gated_delta_state(config)?;
    let mut fused_state = backend.prepare_gated_delta_state(config)?;
    let mut convolved = backend.inner.pool.allocate(&backend.inner.stream, TOKENS * channels)?;
    let key_elements = TOKENS * KEY_HEADS * KEY_DIM;
    let value_elements = TOKENS * VALUE_HEADS * VALUE_DIM;
    let mut split_query = backend.inner.pool.allocate(&backend.inner.stream, key_elements)?;
    let mut split_key = backend.inner.pool.allocate(&backend.inner.stream, key_elements)?;
    let mut split_value = backend.inner.pool.allocate(&backend.inner.stream, value_elements)?;
    split_state.convolve_silu(TOKENS, &input, &weight, &mut convolved)?;
    GatedDeltaTransforms::compile(
        &backend.inner.compiler,
        GatedDeltaTransformSpec {
            tokens: TOKENS,
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            key_dim: KEY_DIM,
            value_dim: VALUE_DIM,
            epsilon: 1.0e-6,
            norm_weight_shift: 0.0,
        },
    )?
    .split_normalize(
        &backend.inner.stream,
        &convolved,
        &mut split_query,
        &mut split_key,
        &mut split_value,
    )?;
    let mut fused_query = backend.inner.pool.allocate(&backend.inner.stream, key_elements)?;
    let mut fused_key = backend.inner.pool.allocate(&backend.inner.stream, key_elements)?;
    let mut fused_value = backend.inner.pool.allocate(&backend.inner.stream, value_elements)?;
    fused_state.convolve_silu_split_normalize_strided(
        TOKENS, &input, &weight, &mut fused_query, &mut fused_key, &mut fused_value, channels, 0,
    )?;
    assert_eq!(read(&backend, &fused_query)?, read(&backend, &split_query)?);
    assert_eq!(read(&backend, &fused_key)?, read(&backend, &split_key)?);
    assert_eq!(read(&backend, &fused_value)?, read(&backend, &split_value)?);
    assert_eq!(
        read(&backend, &fused_state.convolution)?,
        read(&backend, &split_state.convolution)?,
    );
    Ok(())
}
