use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use runtime::tuning::select_fastest_candidate;

use super::MxFp4GatheredMoeBf16;
use crate::{
    CudaBackend, ExecutionPhase, GatedActivation, PlanSource, Result,
    backend::{
        linear::mxfp4::MxFp4ExpertWeights,
        tuning::{MoeProfileExecution, MoeProfileRequest, MxFp4MoeExecution},
    },
};

pub(super) fn prepare(
    backend: &CudaBackend,
    tokens: usize,
    selected_count: usize,
    activation: GatedActivation,
    weights: &MxFp4ExpertWeights,
) -> Result<MxFp4GatheredMoeBf16> {
    let (experts, hidden, intermediate) = weights.geometry();
    let request = MoeProfileRequest::mxfp4(
        if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        },
        tokens,
        experts,
        selected_count,
        hidden,
        intermediate,
        weights.storage(),
        activation,
    );
    if let Some((MoeProfileExecution::MxFp4(execution), source)) =
        backend.auto_tuner().lookup_moe(request)
    {
        trace(request, execution, source, None);
        return MxFp4GatheredMoeBf16::with_candidates(
            backend,
            tokens,
            selected_count,
            activation,
            weights,
            &[execution],
        );
    }
    let fallback = MxFp4MoeExecution::EightWarps;
    if !backend.auto_tuner().claim_moe(request) {
        return MxFp4GatheredMoeBf16::with_candidates(
            backend,
            tokens,
            selected_count,
            activation,
            weights,
            &[fallback],
        );
    }
    match tune(backend, request, tokens, selected_count, activation, weights) {
        Ok(plan) => Ok(plan),
        Err(error) => {
            backend.auto_tuner().abandon_moe(request);
            tracing::warn!(?request, %error, "CUDA gathered MXFP4 tuning failed");
            MxFp4GatheredMoeBf16::with_candidates(
                backend,
                tokens,
                selected_count,
                activation,
                weights,
                &[fallback],
            )
        },
    }
}

fn tune(
    backend: &CudaBackend,
    request: MoeProfileRequest,
    tokens: usize,
    selected_count: usize,
    activation: GatedActivation,
    weights: &MxFp4ExpertWeights,
) -> Result<MxFp4GatheredMoeBf16> {
    let started = Instant::now();
    let executions = [MxFp4MoeExecution::EightWarps, MxFp4MoeExecution::SingleWarp];
    let mut plan = MxFp4GatheredMoeBf16::with_candidates(
        backend, tokens, selected_count, activation, weights, &executions,
    )?;
    let (experts, hidden, _) = weights.geometry();
    let input = sample_input(backend, tokens * hidden)?;
    let selected = sample_selected(backend, tokens * selected_count, experts)?;
    let routing = sample_routing(backend, tokens * selected_count, selected_count)?;
    let mut output = backend.pool().allocate(backend.stream(), tokens * hidden)?;
    validate(backend, &mut plan, &input, &selected, &routing, weights, &mut output)?;
    let (warmup, iterations) = backend.auto_tuner().iterations(tokens);
    let mut timings = Vec::with_capacity(plan.candidates.len());
    for index in 0..plan.candidates.len() {
        for _ in 0..warmup {
            plan.execute_candidate(index, &input, &selected, &routing, weights, &mut output)?;
        }
        let timer = backend.start_device_timer()?;
        for _ in 0..iterations {
            plan.execute_candidate(index, &input, &selected, &routing, weights, &mut output)?;
        }
        timings.push(timer.finish(backend)? / iterations);
    }
    let fastest = timings
        .iter()
        .enumerate()
        .min_by_key(|(_, timing)| **timing)
        .map(|(index, _)| index)
        .ok_or(crate::Error::InvalidExecutionPlan("MXFP4 tuner has no candidates"))?;
    let selected_index = select_fastest_candidate(
        fastest,
        0,
        &timings,
        backend.auto_tuner().minimum_improvement_bps(),
    );
    let execution = plan.candidates[selected_index].execution;
    let average = timings[selected_index];
    plan.retain(selected_index);
    backend.auto_tuner().record_moe(
        request,
        MoeProfileExecution::MxFp4(execution),
        average,
        started.elapsed(),
    );
    trace(request, execution, PlanSource::MeasuredStartup, Some(average));
    Ok(plan)
}

fn validate(
    backend: &CudaBackend,
    plan: &mut MxFp4GatheredMoeBf16,
    input: &DeviceBuffer<bf16>,
    selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>,
    weights: &MxFp4ExpertWeights,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    plan.execute_candidate(0, input, selected, routing, weights, output)?;
    let expected = read(backend, output)?;
    for index in 1..plan.candidates.len() {
        plan.execute_candidate(index, input, selected, routing, weights, output)?;
        if read(backend, output)? != expected {
            return Err(crate::Error::InvalidExecutionPlan("MXFP4 candidates differ"));
        }
    }
    Ok(())
}

fn sample_input(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    const PATTERN: [f32; 17] = [
        -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 0.0, 0.0625, 0.125,
        0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5,
    ];
    let values = (0..elements)
        .map(|index| bf16::from_f32(PATTERN[index % PATTERN.len()]))
        .collect::<Vec<_>>();
    copy(backend, &values)
}

fn sample_selected(
    backend: &CudaBackend,
    assignments: usize,
    experts: usize,
) -> Result<DeviceBuffer<u32>> {
    let values = (0..assignments)
        .map(|index| u32::try_from(index % experts))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    copy(backend, &values)
}

fn sample_routing(
    backend: &CudaBackend,
    assignments: usize,
    selected_count: usize,
) -> Result<DeviceBuffer<bf16>> {
    let divisor = f32::from(u16::try_from(selected_count)?);
    copy(backend, &vec![bf16::from_f32(1.0 / divisor); assignments])
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, values: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    backend.stream().copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}

fn trace(
    request: MoeProfileRequest,
    execution: MxFp4MoeExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?request, ?execution, ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA gathered MXFP4 MoE execution"
    );
}
