use std::time::Duration;

use mircuda::{DeviceBuffer, bf16};
use runtime::tuning::select_fastest_candidate;

use super::{Candidate, CudaBackend, DirectFp8CheckpointWeight, DirectFp8Spec, Error, Result};
use crate::{backend::tuning::QuantizedProfileRequest, kernels::CacheEviction};

const MAX_BF16_ULPS: u16 = 1;
const VALIDATION_ELEMENTS: usize = 4_096;

#[allow(clippy::too_many_arguments)]
pub(super) fn retain_compatible(
    backend: &CudaBackend,
    request: QuantizedProfileRequest,
    candidates: &mut Vec<Candidate>,
    input: &DeviceBuffer<bf16>,
    weight: &DirectFp8CheckpointWeight,
    identity_scale: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    candidates[0].execute(backend.stream(), input, weight, identity_scale, output)?;
    let reference = read_sample(backend, output)?;
    let mut accepted = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let compatible = index == 0 || {
            candidate.execute(backend.stream(), input, weight, identity_scale, output)?;
            equivalent(&reference, &read_sample(backend, output)?)
        };
        if !compatible {
            tracing::warn!(
                ?request,
                execution = ?candidate.execution,
                "rejected numerically incompatible direct FP8 candidate"
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
        .ok_or(Error::InvalidExecutionPlan("direct FP8 tuner rejected its fallback"))
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
            reference.to_f32().is_finite()
                && candidate.to_f32().is_finite()
                && bf16_ulp_distance(*reference, *candidate) <= MAX_BF16_ULPS
        })
}

fn bf16_ulp_distance(left: bf16, right: bf16) -> u16 {
    let left_bits = left.to_bits();
    let right_bits = right.to_bits();
    if left_bits == right_bits || (zero_bits(left_bits) && zero_bits(right_bits)) {
        return 0;
    }
    ordered_bf16(left).abs_diff(ordered_bf16(right))
}

const fn zero_bits(bits: u16) -> bool {
    matches!(bits, 0 | 0x8000)
}

fn ordered_bf16(value: bf16) -> u16 {
    let bits = value.to_bits();
    if bits & 0x8000 == 0 {
        bits | 0x8000
    } else {
        !bits
    }
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub(super) fn select(
    backend: &CudaBackend,
    spec: DirectFp8Spec,
    candidates: &mut [Candidate],
    input: &DeviceBuffer<bf16>,
    weight: &DirectFp8CheckpointWeight,
    identity_scale: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<bf16>,
) -> Result<(usize, Duration, Duration)> {
    let (warmup, iterations) = backend.auto_tuner().iterations(spec.tokens);
    let eviction = CacheEviction::compile(backend.compiler(), backend.pool(), backend.stream())?;
    let mut timings = Vec::with_capacity(candidates.len());
    let mut elapsed = Duration::ZERO;
    for candidate in candidates {
        let average = measure_candidate(
            backend, candidate, input, weight, identity_scale, output, &eviction, warmup,
            iterations,
        )?;
        elapsed = elapsed.saturating_add(average.saturating_mul(iterations + warmup));
        timings.push(average);
    }
    let fastest = timings
        .iter()
        .enumerate()
        .min_by_key(|(_, duration)| **duration)
        .map(|value| value.0)
        .ok_or(Error::InvalidExecutionPlan("direct FP8 tuner has no candidates"))?;
    let selected = select_fastest_candidate(
        fastest,
        0,
        &timings,
        backend.auto_tuner().minimum_improvement_bps(),
    );
    Ok((selected, timings[selected], elapsed))
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn measure_candidate(
    backend: &CudaBackend,
    candidate: &Candidate,
    input: &DeviceBuffer<bf16>,
    weight: &DirectFp8CheckpointWeight,
    identity_scale: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<bf16>,
    eviction: &CacheEviction,
    warmup: u32,
    iterations: u32,
) -> Result<Duration> {
    for _ in 0..warmup {
        eviction.execute(backend.stream())?;
        candidate.execute(backend.stream(), input, weight, identity_scale, output)?;
    }
    let mut measurements = Vec::with_capacity(usize::try_from(iterations)?);
    for _ in 0..iterations {
        eviction.execute(backend.stream())?;
        let started = backend.context().create_event(true)?;
        let completed = backend.context().create_event(true)?;
        started.record(backend.stream())?;
        candidate.execute(backend.stream(), input, weight, identity_scale, output)?;
        completed.record(backend.stream())?;
        measurements.push((started, completed));
    }
    measurements
        .last()
        .ok_or(Error::InvalidExecutionPlan("direct FP8 tuner has zero iterations"))?
        .1
        .synchronize()?;
    let total_ms = measurements.iter().try_fold(0.0_f32, |total, (started, completed)| {
        Ok::<_, Error>(total + started.elapsed_ms(completed)?)
    })?;
    Ok(Duration::from_secs_f32(total_ms / (iterations as f32 * 1_000.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_gate_accepts_one_bf16_ulp_and_rejects_two() {
        let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
        let close = reference.map(|value| bf16::from_bits(value.to_bits() + 1));
        let drift = reference.map(|value| bf16::from_bits(value.to_bits() + 2));
        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
