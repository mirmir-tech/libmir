mod fixture;

use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::super::*;
use crate::{
    CudaConfig, ExecutionPhase, PlanSource, Result,
    backend::tuning::{MoeProfileExecution, MoeProfileRequest},
};

#[test]
fn executes_affine_shared_and_routed_experts_for_prefill_and_decode() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = config();
    let fixture = fixture::MoeFixture::new(config)?;
    let tensors = fixture.upload(&backend)?;
    let moe = CudaAffineSharedExpertMoe::from_tensors(&backend, &tensors, fixture::PREFIX, config)?;

    let input = copy(&backend, &vec![bf16::ZERO; 2 * config.hidden_size])?;
    let mut output = allocate(&backend, 2 * config.hidden_size)?;
    moe.prepare(2)?.execute(&input, &mut output)?;
    assert!(read(&backend, &output)?.iter().all(|value| *value == bf16::ZERO));
    measured(&backend, config, ExecutionPhase::Prefill, 2);

    let input = copy(&backend, &vec![bf16::ZERO; config.hidden_size])?;
    let mut output = allocate(&backend, config.hidden_size)?;
    moe.prepare(1)?.execute(&input, &mut output)?;
    assert!(read(&backend, &output)?.iter().all(|value| *value == bf16::ZERO));
    measured(&backend, config, ExecutionPhase::Decode, 1);
    Ok(())
}

fn measured(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    phase: ExecutionPhase,
    tokens: usize,
) {
    let request = MoeProfileRequest::affine(
        phase,
        tokens,
        config.expert_count,
        config.top_k,
        config.hidden_size,
        config.routed_intermediate_size,
        config.group_size,
        config.expert_bits,
        config.activation,
    );
    assert!(matches!(
        backend.auto_tuner().lookup_moe(request),
        Some((MoeProfileExecution::Affine(_), PlanSource::MeasuredStartup))
    ));
}

fn config() -> AffineSharedExpertMoeConfig {
    AffineSharedExpertMoeConfig {
        hidden_size: 64,
        routed_intermediate_size: 64,
        shared_intermediate_size: 64,
        expert_count: 3,
        top_k: 2,
        group_size: 64,
        expert_bits: 4,
        router_bits: 8,
        activation: GatedActivation::Silu,
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
