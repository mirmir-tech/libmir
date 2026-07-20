mod fixture;

use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::super::*;
use crate::{CudaConfig, Result};

#[test]
fn executes_complete_affine_gated_delta_prefill_and_decode() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = config();
    let fixture = fixture::LayerFixture::new(config)?;
    let tensors = fixture.upload(&backend)?;
    let layer =
        CudaAffineGatedDeltaLayer::from_tensors(&backend, &tensors, fixture::PREFIX, config)?;
    let mut state = layer.prepare_state()?;

    let input = copy(&backend, &vec![bf16::ZERO; 2 * config.hidden_size])?;
    let mut output = allocate(&backend, 2 * config.hidden_size)?;
    layer.prepare(2)?.execute(&input, &mut state, &mut output)?;
    assert!(read(&backend, &output)?.iter().all(|value| *value == bf16::ZERO));
    assert_eq!(state.offset(), 2);

    let input = copy(&backend, &vec![bf16::ZERO; config.hidden_size])?;
    let mut output = allocate(&backend, config.hidden_size)?;
    layer.prepare(1)?.execute(&input, &mut state, &mut output)?;
    assert!(read(&backend, &output)?.iter().all(|value| *value == bf16::ZERO));
    assert_eq!(state.offset(), 3);
    Ok(())
}

fn config() -> AffineGatedDeltaLayerConfig {
    AffineGatedDeltaLayerConfig {
        hidden_size: 64,
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 64,
        convolution_kernel_size: 2,
        group_size: 64,
        bits: 4,
        rms_norm_epsilon: 1.0e-6,
        norm_weight_shift: 0.0,
    }
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
