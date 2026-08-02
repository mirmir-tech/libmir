use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, bf16};
use runtime::tuning::select_fastest_candidate;

use super::{AutoAffineRoutedExperts, candidate::Candidate};
use crate::{
    PlanSource, Result,
    backend::{
        shared_moe::weights::AffineRoutedMoeWeights,
        tuning::{AffineMoeExecution, MoeProfileExecution, MoeProfileRequest},
    },
};

const ABSOLUTE_TOLERANCE: f32 = 0.125;
const RELATIVE_TOLERANCE: f32 = 0.01;

impl AutoAffineRoutedExperts {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tune(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &AffineRoutedMoeWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let started = Instant::now();
        self.prepare_candidates();
        self.retain_compatible(input, selected, routing, weights, intermediate, output)?;
        let (winner, average, measured) =
            self.measure(input, selected, routing, weights, intermediate, output)?;
        let execution = self.candidates[winner].execution;
        self.retain(winner);
        self.backend.auto_tuner().record_moe(
            self.profile,
            MoeProfileExecution::Affine(execution),
            average,
            started.elapsed().max(measured),
        );
        trace_selection(self.profile, execution, PlanSource::MeasuredStartup, Some(average));
        Ok(())
    }

    fn prepare_candidates(&mut self) {
        for execution in [AffineMoeExecution::FusedGated, AffineMoeExecution::SeparatePair] {
            if self.candidates.iter().any(|candidate| candidate.execution == execution) {
                continue;
            }
            match Candidate::new(&self.backend, self.config, self.tokens, execution) {
                Ok(candidate) => self.candidates.push(candidate),
                Err(error) => tracing::debug!(
                    ?execution,
                    %error,
                    "discarded unavailable affine MoE tuning candidate"
                ),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn retain_compatible(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &AffineRoutedMoeWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.candidates[self.fallback]
            .execute(input, selected, routing, weights, intermediate, output)?;
        let context = self.backend.context().clone();
        let stream = self.backend.stream().clone();
        let reference = read(&context, &stream, output)?;
        let mut accepted = Vec::with_capacity(self.candidates.len());
        for (index, candidate) in self.candidates.iter_mut().enumerate() {
            let compatible = if index == self.fallback {
                true
            } else {
                candidate.execute(input, selected, routing, weights, intermediate, output)?;
                equivalent(&reference, &read(&context, &stream, output)?)
            };
            if !compatible {
                tracing::warn!(
                    execution = ?candidate.execution,
                    "rejected numerically incompatible affine MoE candidate"
                );
            }
            accepted.push(compatible);
        }
        self.filter_candidates(&accepted)
    }

    fn filter_candidates(&mut self, accepted: &[bool]) -> Result<()> {
        let mut retained = Vec::with_capacity(self.candidates.len());
        let mut fallback = None;
        for (index, candidate) in self.candidates.drain(..).enumerate() {
            if accepted[index] {
                if index == self.fallback {
                    fallback = Some(retained.len());
                }
                retained.push(candidate);
            }
        }
        self.candidates = retained;
        self.fallback = fallback
            .ok_or(crate::Error::InvalidExecutionPlan("affine MoE tuner rejected its fallback"))?;
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
    fn measure(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &AffineRoutedMoeWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<(usize, Duration, Duration)> {
        let (warmup, iterations) = self.backend.auto_tuner().iterations(self.tokens);
        let mut timings = Vec::with_capacity(self.candidates.len());
        let mut elapsed = Duration::ZERO;
        for candidate in &mut self.candidates {
            for _ in 0..warmup {
                candidate.execute(input, selected, routing, weights, intermediate, output)?;
            }
            let started = self.backend.context().create_event(true)?;
            let completed = self.backend.context().create_event(true)?;
            started.record(self.backend.stream())?;
            for _ in 0..iterations {
                candidate.execute(input, selected, routing, weights, intermediate, output)?;
            }
            completed.record(self.backend.stream())?;
            completed.synchronize()?;
            let average = Duration::from_secs_f32(
                started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
            );
            elapsed =
                elapsed.saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
            timings.push(average);
        }
        let fastest = timings
            .iter()
            .enumerate()
            .min_by_key(|(_, duration)| **duration)
            .map(|value| value.0)
            .ok_or(crate::Error::InvalidExecutionPlan("affine MoE tuner has no candidates"))?;
        let selected = select_fastest_candidate(
            fastest,
            self.fallback,
            &timings,
            self.backend.auto_tuner().minimum_improvement_bps(),
        );
        Ok((selected, timings[selected], elapsed))
    }
}

fn read(
    context: &mircuda::Context,
    stream: &mircuda::Stream,
    output: &DeviceBuffer<bf16>,
) -> Result<Vec<bf16>> {
    let mut host = context.allocate_pinned(output.len())?;
    stream.copy_to_host(output, &mut host)?;
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

pub(super) fn trace_selection(
    request: MoeProfileRequest,
    execution: AffineMoeExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?request,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA affine routed-MoE execution"
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
