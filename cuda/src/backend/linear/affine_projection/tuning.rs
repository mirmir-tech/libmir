use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, bf16};
use runtime::tuning::select_fastest_candidate;

use super::{AffineQuantizedConfig, AffineQuantizedWeight, Candidate, CudaBackend};
use crate::{
    Error, PlanSource, Result,
    backend::tuning::{
        AffineProjectionExecution, QuantizedProfileExecution, QuantizedProfileRequest,
    },
};

const ABSOLUTE_TOLERANCE: f32 = 0.125;
const RELATIVE_TOLERANCE: f32 = 0.01;

pub(super) fn prepare(
    backend: &CudaBackend,
    request: QuantizedProfileRequest,
    tokens: usize,
    config: AffineQuantizedConfig,
    weights: &AffineQuantizedWeight,
) -> Result<Candidate> {
    if let Some((QuantizedProfileExecution::Affine(execution), source)) =
        backend.auto_tuner().lookup_quantized(request)
    {
        match Candidate::new(backend, tokens, config, execution) {
            Ok(candidate) => {
                trace_selection(request, execution, source, None);
                return Ok(candidate);
            },
            Err(error) => tracing::warn!(
                ?execution,
                %error,
                "cached affine projection candidate is unavailable; using QMM"
            ),
        }
    }
    let fallback = Candidate::new(backend, tokens, config, AffineProjectionExecution::Qmm)?;
    if tokens != 1 || !backend.auto_tuner().claim_quantized(request) {
        return Ok(fallback);
    }
    match tune(backend, request, config, weights, fallback) {
        Ok(candidate) => Ok(candidate),
        Err(error) => {
            backend.auto_tuner().abandon_quantized(request);
            tracing::warn!(
                ?request,
                %error,
                "CUDA affine projection tuning failed; retaining QMM"
            );
            Candidate::new(backend, tokens, config, AffineProjectionExecution::Qmm)
        },
    }
}

fn tune(
    backend: &CudaBackend,
    request: QuantizedProfileRequest,
    config: AffineQuantizedConfig,
    weights: &AffineQuantizedWeight,
    fallback: Candidate,
) -> Result<Candidate> {
    let started = Instant::now();
    let mut candidates = vec![fallback];
    match Candidate::new(backend, 1, config, AffineProjectionExecution::Gemv) {
        Ok(candidate) => candidates.push(candidate),
        Err(error) => tracing::debug!(%error, "affine GEMV tuning candidate is unavailable"),
    }
    let input = sample_input(backend, config.input_features)?;
    let mut output = backend
        .pool()
        .allocate_zeroed::<bf16>(backend.stream(), config.output_features)?;
    retain_compatible(backend, request, &mut candidates, &input, weights, &mut output)?;
    let (selected, average, measured) =
        measure(backend, &mut candidates, &input, weights, &mut output)?;
    let selected = candidates.swap_remove(selected);
    backend.auto_tuner().record_quantized(
        request,
        QuantizedProfileExecution::Affine(selected.execution),
        average,
        started.elapsed().max(measured),
    );
    trace_selection(request, selected.execution, PlanSource::MeasuredStartup, Some(average));
    Ok(selected)
}

fn sample_input(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    const PATTERN: [f32; 17] = [
        -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 0.0, 0.0625, 0.125,
        0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5,
    ];
    let values = (0..elements)
        .map(|index| bf16::from_f32(PATTERN[index % PATTERN.len()]))
        .collect::<Vec<_>>();
    let mut host = backend.context().allocate_pinned(elements)?;
    host.copy_from_slice(&values)?;
    let mut input = backend.pool().allocate(backend.stream(), elements)?;
    backend.stream().copy_to_device(&mut host, &mut input)?;
    Ok(input)
}

fn retain_compatible(
    backend: &CudaBackend,
    request: QuantizedProfileRequest,
    candidates: &mut Vec<Candidate>,
    input: &DeviceBuffer<bf16>,
    weights: &AffineQuantizedWeight,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    candidates[0].execute(input, weights, output)?;
    let reference = read(backend, output)?;
    let mut accepted = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let compatible = index == 0 || {
            candidate.execute(input, weights, output)?;
            equivalent(&reference, &read(backend, output)?)
        };
        if !compatible {
            tracing::warn!(
                ?request,
                execution = ?candidate.execution,
                "rejected numerically incompatible affine projection candidate"
            );
        }
        accepted.push(compatible);
    }
    let mut index = 0;
    candidates.retain(|_| {
        let keep = accepted[index];
        index += 1;
        keep
    });
    (!candidates.is_empty())
        .then_some(())
        .ok_or(Error::InvalidExecutionPlan("affine projection tuner rejected QMM"))
}

fn read(backend: &CudaBackend, output: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.context().allocate_pinned(output.len())?;
    backend.stream().copy_to_host(output, &mut host)?;
    Ok(host.to_vec()?)
}

fn equivalent(reference: &[bf16], candidate: &[bf16]) -> bool {
    reference.len() == candidate.len()
        && reference.iter().zip(candidate).all(|(reference, candidate)| {
            let reference = reference.to_f32();
            let candidate = candidate.to_f32();
            reference.is_finite()
                && candidate.is_finite()
                && (reference - candidate).abs()
                    <= ABSOLUTE_TOLERANCE.max(reference.abs() * RELATIVE_TOLERANCE)
        })
}

#[allow(clippy::cast_precision_loss)]
fn measure(
    backend: &CudaBackend,
    candidates: &mut [Candidate],
    input: &DeviceBuffer<bf16>,
    weights: &AffineQuantizedWeight,
    output: &mut DeviceBuffer<bf16>,
) -> Result<(usize, Duration, Duration)> {
    let (warmup, iterations) = backend.auto_tuner().iterations(1);
    let mut timings = Vec::with_capacity(candidates.len());
    let mut elapsed = Duration::ZERO;
    for candidate in candidates {
        for _ in 0..warmup {
            candidate.execute(input, weights, output)?;
        }
        let started = backend.context().create_event(true)?;
        let completed = backend.context().create_event(true)?;
        started.record(backend.stream())?;
        for _ in 0..iterations {
            candidate.execute(input, weights, output)?;
        }
        completed.record(backend.stream())?;
        completed.synchronize()?;
        let average = Duration::from_secs_f32(
            started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
        );
        elapsed = elapsed.saturating_add(average.saturating_mul(iterations + warmup));
        timings.push(average);
    }
    let fastest = timings
        .iter()
        .enumerate()
        .min_by_key(|(_, duration)| **duration)
        .map(|value| value.0)
        .ok_or(Error::InvalidExecutionPlan("affine projection tuner has no candidates"))?;
    let selected = select_fastest_candidate(
        fastest,
        0,
        &timings,
        backend.auto_tuner().minimum_improvement_bps(),
    );
    Ok((selected, timings[selected], elapsed))
}

fn trace_selection(
    request: QuantizedProfileRequest,
    execution: AffineProjectionExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?request,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA affine projection execution"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_gate_accepts_rounding_and_rejects_drift() {
        let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
        let close = [1.125, -20.125, 0.125].map(bf16::from_f32);
        let drift = [1.5, -20.0, 0.0].map(bf16::from_f32);
        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
