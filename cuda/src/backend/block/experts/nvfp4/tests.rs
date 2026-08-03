use mircuda::{DeviceBuffer, DeviceElement};

use super::AutoNvFp4Experts;
use crate::{
    CudaBackend, CudaConfig, ExecutionPhase, GatedActivation, MoePlanRequest, NvFp4ExpertBank,
    NvFp4ExpertBankConfig, PlanSource, Result,
    backend::tuning::{MoeProfileExecution, MoeProfileRequest},
    kernels::{RoutePattern, RoutePatternGenerator, RoutePatternSpec},
};

const EXPERTS: usize = 8;
const SELECTED: usize = 4;
const HIDDEN: usize = 2_816;
const INTERMEDIATE: usize = 704;

#[test]
fn synthetic_nvfp4_autotunes_decode_and_prefill() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let gate = NvFp4ExpertBank::synthetic_zero(&backend, bank(HIDDEN, INTERMEDIATE))?;
    let up = NvFp4ExpertBank::synthetic_zero(&backend, bank(HIDDEN, INTERMEDIATE))?;
    let down = NvFp4ExpertBank::synthetic_zero(&backend, bank(INTERMEDIATE, HIDDEN))?;

    run(&backend, ExecutionPhase::Decode, 1, &gate, &up, &down)?;
    run(&backend, ExecutionPhase::Prefill, 4, &gate, &up, &down)?;
    run_weight_only(&backend, ExecutionPhase::Decode, 1, &gate, &up, &down)?;
    run_weight_only(&backend, ExecutionPhase::Prefill, 4, &gate, &up, &down)?;
    Ok(())
}

fn run(
    backend: &CudaBackend,
    phase: ExecutionPhase,
    tokens: usize,
    gate: &NvFp4ExpertBank,
    up: &NvFp4ExpertBank,
    down: &NvFp4ExpertBank,
) -> Result<()> {
    let mut experts = AutoNvFp4Experts::new(
        backend,
        phase,
        tokens,
        SELECTED,
        GatedActivation::GeluTanh,
        gate.clone(),
        up.clone(),
        down.clone(),
        models::weights::BlockActivationMode::WeightAndActivation,
    )?;
    let input = backend.pool().allocate_zeroed(backend.stream(), tokens * HIDDEN)?;
    let expert_count = u32::try_from(EXPERTS)?;
    let selection_count = u32::try_from(tokens * SELECTED)?;
    let selected_values =
        (0..selection_count).map(|expert| expert % expert_count).collect::<Vec<_>>();
    let selected = copy(backend, &selected_values)?;
    let routing = backend.pool().allocate_zeroed(backend.stream(), tokens * SELECTED)?;
    let mut output = backend.pool().allocate_zeroed(backend.stream(), tokens * HIDDEN)?;

    experts.execute(&input, &selected, &routing, &mut output)?;
    backend.synchronize()?;
    let request = MoePlanRequest::nvfp4(phase, tokens, EXPERTS, SELECTED, HIDDEN, INTERMEDIATE);
    let profile = MoeProfileRequest::nvfp4(request, GatedActivation::GeluTanh, false);
    assert!(matches!(
        backend.auto_tuner().lookup_moe(profile),
        Some((MoeProfileExecution::NvFp4(_), PlanSource::MeasuredStartup))
    ));
    Ok(())
}

fn run_weight_only(
    backend: &CudaBackend,
    phase: ExecutionPhase,
    tokens: usize,
    gate: &NvFp4ExpertBank,
    up: &NvFp4ExpertBank,
    down: &NvFp4ExpertBank,
) -> Result<()> {
    let mut experts = AutoNvFp4Experts::new(
        backend,
        phase,
        tokens,
        SELECTED,
        GatedActivation::GeluTanh,
        gate.clone(),
        up.clone(),
        down.clone(),
        models::weights::BlockActivationMode::WeightOnly,
    )?;
    let input = backend.pool().allocate_zeroed(backend.stream(), tokens * HIDDEN)?;
    let selected = backend.pool().allocate_zeroed(backend.stream(), tokens * SELECTED)?;
    let routing = backend.pool().allocate_zeroed(backend.stream(), tokens * SELECTED)?;
    let mut output = backend.pool().allocate_zeroed(backend.stream(), tokens * HIDDEN)?;
    experts.execute(&input, &selected, &routing, &mut output)?;
    backend.synchronize()?;
    let request = MoePlanRequest::nvfp4(phase, tokens, EXPERTS, SELECTED, HIDDEN, INTERMEDIATE);
    let profile = MoeProfileRequest::nvfp4(request, GatedActivation::GeluTanh, true);
    assert!(matches!(
        backend.auto_tuner().lookup_moe(profile),
        Some((MoeProfileExecution::NvFp4(_), PlanSource::MeasuredStartup))
    ));
    Ok(())
}

#[test]
fn synthetic_route_patterns_cover_balanced_and_hot_set_extremes() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = RoutePatternSpec { tokens: 3, experts: 8, top_k: 2 };
    let generator = RoutePatternGenerator::compile(backend.compiler(), spec)?;
    let mut selected = backend.pool().allocate(backend.stream(), 6)?;

    generator.execute(backend.stream(), RoutePattern::Balanced, &mut selected)?;
    assert_eq!(read(backend.context(), backend.stream(), &selected)?, [0, 1, 2, 3, 4, 5]);
    generator.execute(backend.stream(), RoutePattern::HotSet, &mut selected)?;
    assert_eq!(read(backend.context(), backend.stream(), &selected)?, [0, 1, 0, 1, 0, 1]);
    Ok(())
}

fn bank(input_features: usize, output_features: usize) -> NvFp4ExpertBankConfig {
    NvFp4ExpertBankConfig {
        experts: EXPERTS,
        input_features,
        output_features,
    }
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.context().allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement + Copy>(
    context: &mircuda::Context,
    stream: &mircuda::Stream,
    values: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = context.allocate_pinned(values.len())?;
    stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
