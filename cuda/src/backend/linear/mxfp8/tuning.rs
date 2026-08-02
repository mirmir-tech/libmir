use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, MxFp8Spec, bf16};
use runtime::tuning::select_fastest_candidate;

use super::{
    MxFp8CheckpointWeight,
    candidate::{Candidate, tensor_core_admitted},
};
use crate::{
    CudaBackend, Error, PlanSource, Result,
    backend::tuning::{
        MxFp8ProjectionExecution, QuantizedProfileExecution, QuantizedProfileRequest,
    },
};

const ABSOLUTE_TOLERANCE: f32 = 0.25;
const RELATIVE_TOLERANCE: f32 = 0.02;
const VALIDATION_ELEMENTS: usize = 4_096;

pub(super) fn prepare(
    backend: &CudaBackend,
    spec: MxFp8Spec,
    weight: &MxFp8CheckpointWeight,
) -> Result<Candidate> {
    let request = QuantizedProfileRequest::mxfp8(
        spec.tokens(),
        spec.input_features(),
        spec.output_features(),
    );
    if let Some((QuantizedProfileExecution::MxFp8(execution), source)) =
        backend.auto_tuner().lookup_quantized(request)
    {
        match Candidate::new(backend, spec, weight, execution) {
            Ok(candidate) => {
                trace_selection(request, execution, source, None);
                return Ok(candidate);
            },
            Err(error) => tracing::warn!(
                ?execution,
                %error,
                "cached MXFP8 candidate is unavailable; using the stable fallback"
            ),
        }
    }
    let fallback_execution = if tensor_core_admitted(backend, spec) {
        MxFp8ProjectionExecution::TensorCore
    } else {
        MxFp8ProjectionExecution::Portable
    };
    let fallback = Candidate::new(backend, spec, weight, fallback_execution)?;
    if !tensor_core_admitted(backend, spec) || !backend.auto_tuner().claim_quantized(request) {
        return Ok(fallback);
    }
    match tune(backend, request, spec, weight, fallback) {
        Ok(candidate) => Ok(candidate),
        Err(error) => {
            backend.auto_tuner().abandon_quantized(request);
            tracing::warn!(?request, %error, "CUDA MXFP8 tuning failed; retaining Tensor Core");
            Candidate::new(backend, spec, weight, fallback_execution)
        },
    }
}

fn tune(
    backend: &CudaBackend,
    request: QuantizedProfileRequest,
    spec: MxFp8Spec,
    weight: &MxFp8CheckpointWeight,
    fallback: Candidate,
) -> Result<Candidate> {
    let started = Instant::now();
    let mut candidates = vec![fallback];
    match Candidate::new(backend, spec, weight, MxFp8ProjectionExecution::Portable) {
        Ok(candidate) => candidates.push(candidate),
        Err(error) => tracing::debug!(%error, "portable MXFP8 tuning candidate is unavailable"),
    }
    let input = sample_input(backend, spec.tokens() * spec.input_features())?;
    let mut output = backend
        .pool()
        .allocate_zeroed::<bf16>(backend.stream(), spec.tokens() * spec.output_features())?;
    retain_compatible(backend, request, &mut candidates, &input, weight, &mut output)?;
    let (selected, average, measured) =
        measure(backend, spec, &mut candidates, &input, weight, &mut output)?;
    let selected = candidates.swap_remove(selected);
    backend.auto_tuner().record_quantized(
        request,
        QuantizedProfileExecution::MxFp8(selected.execution),
        average,
        started.elapsed().max(measured),
    );
    trace_selection(request, selected.execution, PlanSource::MeasuredStartup, Some(average));
    Ok(selected)
}

fn sample_input(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    const PATTERN: [f32; 16] = [
        -1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, -0.03125, 0.0, 0.03125, 0.0625, 0.125, 0.25,
        0.5, 0.75, 1.0, 0.375,
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
    weight: &MxFp8CheckpointWeight,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    candidates[0].execute(backend.stream(), backend.pool(), input, weight, output)?;
    let reference = read_sample(backend, output)?;
    let mut accepted = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let compatible = index == 0 || {
            candidate.execute(backend.stream(), backend.pool(), input, weight, output)?;
            equivalent(&reference, &read_sample(backend, output)?)
        };
        if !compatible {
            tracing::warn!(
                ?request,
                execution = ?candidate.execution,
                "rejected numerically incompatible MXFP8 projection candidate"
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
        .ok_or(Error::InvalidExecutionPlan("MXFP8 tuner rejected its fallback"))
}

fn read_sample(backend: &CudaBackend, output: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let elements = output.len().min(VALIDATION_ELEMENTS);
    let mut sample = backend.pool().allocate::<bf16>(backend.stream(), elements)?;
    backend.stream().copy_device_range(output, 0..elements, &mut sample, 0)?;
    let mut host = backend.context().allocate_pinned(elements)?;
    backend.stream().copy_to_host(&sample, &mut host)?;
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
    spec: MxFp8Spec,
    candidates: &mut [Candidate],
    input: &DeviceBuffer<bf16>,
    weight: &MxFp8CheckpointWeight,
    output: &mut DeviceBuffer<bf16>,
) -> Result<(usize, Duration, Duration)> {
    let (warmup, iterations) = backend.auto_tuner().iterations(spec.tokens());
    let mut timings = Vec::with_capacity(candidates.len());
    let mut elapsed = Duration::ZERO;
    for candidate in candidates {
        for _ in 0..warmup {
            candidate.execute(backend.stream(), backend.pool(), input, weight, output)?;
        }
        let started = backend.context().create_event(true)?;
        let completed = backend.context().create_event(true)?;
        started.record(backend.stream())?;
        for _ in 0..iterations {
            candidate.execute(backend.stream(), backend.pool(), input, weight, output)?;
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
        .ok_or(Error::InvalidExecutionPlan("MXFP8 tuner has no candidates"))?;
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
    execution: MxFp8ProjectionExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?request,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA MXFP8 projection execution"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_gate_accepts_bounded_quantization_and_rejects_drift() {
        let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
        let close = [1.25, -20.25, 0.25].map(bf16::from_f32);
        let drift = [1.5, -20.0, 0.0].map(bf16::from_f32);
        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
