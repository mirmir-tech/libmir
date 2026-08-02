use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::AutoClampedExperts;
use crate::{
    CudaBackend, CudaConfig, CudaTensor, ExecutionPhase, PlanSource, Result,
    backend::{
        clamped_routed::{
            ClampedRoutedConfig,
            weights::{ClampedRoutedExpertWeights, NativeExpertWeights},
        },
        tuning::{ClampedMoeStorage, MoeProfileExecution, MoeProfileRequest},
    },
    kernels::{ClampedRoutedKernels, ClampedRoutedSpec},
};

const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const WIDTH: usize = 32;

#[test]
fn synthetic_clamped_experts_autotune_decode_and_prefill() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let weights = native_weights(&backend)?;
    run(&backend, &weights, ExecutionPhase::Decode, 1)?;
    run(&backend, &weights, ExecutionPhase::Prefill, 2)
}

fn run(
    backend: &CudaBackend,
    weights: &ClampedRoutedExpertWeights,
    phase: ExecutionPhase,
    tokens: usize,
) -> Result<()> {
    let config = config();
    let kernels = ClampedRoutedKernels::compile(backend.compiler(), kernel_spec(config, tokens))?;
    let mut experts = AutoClampedExperts::new(backend, config, tokens, phase, weights, kernels)
        .ok_or(crate::Error::InvalidExecutionPlan("missing synthetic clamped experts"))?;
    let input = upload(backend, &vec![bf16::from_f32(1.0); tokens * WIDTH])?;
    let selections = u32::try_from(tokens * TOP_K)?;
    let expert_count = u32::try_from(EXPERTS)?;
    let selected =
        upload(backend, &(0..selections).map(|value| value % expert_count).collect::<Vec<_>>())?;
    let routing = upload(backend, &vec![bf16::from_f32(0.5); tokens * TOP_K])?;
    let mut activated = backend.pool().allocate(backend.stream(), tokens * TOP_K * WIDTH)?;
    let mut partial = backend.pool().allocate(backend.stream(), tokens * TOP_K * WIDTH)?;
    let mut output = backend.pool().allocate(backend.stream(), tokens * WIDTH)?;

    experts
        .execute(weights, &input, &selected, &routing, &mut activated, &mut partial, &mut output)?;
    backend.synchronize()?;
    let profile = MoeProfileRequest::clamped(
        phase,
        tokens,
        EXPERTS,
        TOP_K,
        WIDTH,
        WIDTH,
        ClampedMoeStorage::Native,
    );
    assert!(matches!(
        backend.auto_tuner().lookup_moe(profile),
        Some((MoeProfileExecution::Clamped(_), PlanSource::MeasuredStartup))
    ));
    Ok(())
}

fn native_weights(backend: &CudaBackend) -> Result<ClampedRoutedExpertWeights> {
    let gate_rows = 2 * WIDTH;
    Ok(ClampedRoutedExpertWeights::Native(Box::new(NativeExpertWeights {
        gate_up_blocks: u8_tensor(
            backend,
            "gate-up-blocks",
            vec![EXPERTS, gate_rows, 1, 16],
            0x11,
        )?,
        gate_up_scales: u8_tensor(backend, "gate-up-scales", vec![EXPERTS, gate_rows, 1], 127)?,
        gate_up_bias: bf16_tensor(backend, "gate-up-bias", EXPERTS * gate_rows)?,
        down_blocks: u8_tensor(backend, "down-blocks", vec![EXPERTS, WIDTH, 1, 16], 0x11)?,
        down_scales: u8_tensor(backend, "down-scales", vec![EXPERTS, WIDTH, 1], 127)?,
        down_bias: bf16_tensor(backend, "down-bias", EXPERTS * WIDTH)?,
    })))
}

fn u8_tensor(
    backend: &CudaBackend,
    name: &str,
    shape: Vec<usize>,
    value: u8,
) -> Result<CudaTensor> {
    let elements = shape.iter().product();
    Ok(CudaTensor::from_u8(
        name.into(),
        shape,
        upload(backend, &vec![value; elements])?,
    ))
}

fn bf16_tensor(backend: &CudaBackend, name: &str, elements: usize) -> Result<CudaTensor> {
    Ok(CudaTensor::from_bf16(
        name.into(),
        vec![elements],
        upload(backend, &vec![bf16::from_f32(0.0); elements])?,
    ))
}

fn upload<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn config() -> ClampedRoutedConfig {
    ClampedRoutedConfig {
        vocab: 32,
        hidden: WIDTH,
        intermediate: WIDTH,
        query_heads: 1,
        kv_heads: 1,
        head_dim: WIDTH,
        experts: EXPERTS,
        top_k: TOP_K,
        epsilon: 1.0e-5,
        scale: 1.0,
        theta: 150_000.0,
        factor: 32.0,
        initial_context: 4_096.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        swiglu_limit: 7.0,
    }
}

fn kernel_spec(config: ClampedRoutedConfig, tokens: usize) -> ClampedRoutedSpec {
    ClampedRoutedSpec {
        tokens,
        hidden: config.hidden,
        intermediate: config.intermediate,
        query_heads: config.query_heads,
        kv_heads: config.kv_heads,
        head_dim: config.head_dim,
        top_k: config.top_k,
        theta: config.theta,
        factor: config.factor,
        initial_context: config.initial_context,
        beta_fast: config.beta_fast,
        beta_slow: config.beta_slow,
        swiglu_limit: config.swiglu_limit,
    }
}
