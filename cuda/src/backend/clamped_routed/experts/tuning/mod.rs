use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, bf16};

use super::{
    super::{ClampedRoutedConfig, weights::ClampedRoutedExpertWeights},
    AutoClampedExperts,
};
use crate::{
    ExecutionPhase, PlanSource, Result,
    backend::tuning::{ClampedMoeExecution, MoeProfileExecution},
};

mod measure;

const ABSOLUTE_TOLERANCE: f32 = 0.125;
const RELATIVE_TOLERANCE: f32 = 0.01;
impl AutoClampedExperts {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tune(
        &mut self,
        weights: &ClampedRoutedExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        activated: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let started = Instant::now();
        self.prepare_candidates(weights);
        self.retain_compatible(weights, input, selected, routing, activated, partial, output)?;
        let (winner, average, measured_elapsed) =
            self.measure(weights, input, selected, routing, activated, partial, output)?;
        let execution = self.candidates[winner].execution;
        self.retain(winner);
        self.backend.auto_tuner().record_moe(
            self.profile,
            MoeProfileExecution::Clamped(execution),
            average,
            started.elapsed().max(measured_elapsed),
        );
        trace_selection(
            self.config,
            self.tokens,
            self.phase,
            execution,
            PlanSource::MeasuredStartup,
            Some(average),
        );
        Ok(())
    }

    fn prepare_candidates(&mut self, weights: &ClampedRoutedExpertWeights) {
        for execution in [
            ClampedMoeExecution::FusedReduce,
            ClampedMoeExecution::RouteParallel,
            ClampedMoeExecution::MarlinN128K128,
            ClampedMoeExecution::MarlinN128K64,
            ClampedMoeExecution::MarlinN64K128,
            ClampedMoeExecution::MarlinM64N256K64,
            ClampedMoeExecution::MarlinM64N128K64,
            ClampedMoeExecution::MarlinM64N64K128,
        ]
        .map(|execution| execution.for_batch(self.tokens, self.config.experts, self.config.top_k))
        {
            if self.candidates.iter().any(|candidate| candidate.execution == execution) {
                continue;
            }
            match super::candidate(
                &self.backend, self.config, self.tokens, weights, &self.kernels, execution,
            ) {
                Ok(candidate) => self.candidates.push(candidate),
                Err(error) => tracing::debug!(
                    ?execution,
                    %error,
                    "clamped CUDA MoE candidate is unavailable for this geometry"
                ),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn retain_compatible(
        &mut self,
        weights: &ClampedRoutedExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        activated: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.candidates[self.fallback].execute(
            self.backend.stream(),
            weights,
            input,
            selected,
            routing,
            activated,
            partial,
            output,
        )?;
        let reference = read(&self.backend, output)?;
        let mut accepted = Vec::with_capacity(self.candidates.len());
        for (index, candidate) in self.candidates.iter().enumerate() {
            let compatible = if index == self.fallback {
                true
            } else {
                candidate.execute(
                    self.backend.stream(),
                    weights,
                    input,
                    selected,
                    routing,
                    activated,
                    partial,
                    output,
                )?;
                let comparison = compare(&reference, &read(&self.backend, output)?);
                tracing::debug!(
                    target: "libmir::cuda::tuning",
                    execution = ?candidate.execution,
                    max_abs = comparison.max_abs,
                    max_rel = comparison.max_rel,
                    index = comparison.index,
                    reference = comparison.reference,
                    candidate = comparison.candidate,
                    equivalent = comparison.equivalent,
                    "compared clamped CUDA MoE candidate output"
                );
                if !comparison.equivalent {
                    tracing::warn!(
                        execution = ?candidate.execution,
                        stage = "output",
                        max_abs = comparison.max_abs,
                        max_rel = comparison.max_rel,
                        index = comparison.index,
                        reference = comparison.reference,
                        candidate = comparison.candidate,
                        "clamped CUDA MoE candidate stage differs"
                    );
                }
                comparison.equivalent
            };
            if !compatible {
                tracing::warn!(
                    execution = ?candidate.execution,
                    "rejected numerically incompatible clamped CUDA MoE candidate"
                );
            }
            accepted.push(compatible);
        }
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
            .ok_or(crate::Error::InvalidExecutionPlan("clamped MoE tuner rejected its fallback"))?;
        Ok(())
    }
}

fn read(backend: &crate::CudaBackend, output: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.context().allocate_pinned::<bf16>(output.len())?;
    backend.stream().copy_to_host(output, &mut host)?;
    Ok(host.to_vec()?)
}

struct Comparison {
    equivalent: bool,
    max_abs: f32,
    max_rel: f32,
    index: usize,
    reference: f32,
    candidate: f32,
}

fn compare(reference: &[bf16], candidate: &[bf16]) -> Comparison {
    let mut result = Comparison {
        equivalent: reference.len() == candidate.len(),
        max_abs: 0.0,
        max_rel: 0.0,
        index: 0,
        reference: 0.0,
        candidate: 0.0,
    };
    for (index, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
        let reference = reference.to_f32();
        let candidate = candidate.to_f32();
        let absolute = (reference - candidate).abs();
        let relative = absolute / reference.abs().max(f32::MIN_POSITIVE);
        if !reference.is_finite() || !candidate.is_finite() || absolute > result.max_abs {
            result.max_abs = absolute;
            result.max_rel = relative;
            result.index = index;
            result.reference = reference;
            result.candidate = candidate;
        }
        result.equivalent &= reference.is_finite()
            && candidate.is_finite()
            && absolute <= ABSOLUTE_TOLERANCE.max(reference.abs() * RELATIVE_TOLERANCE);
    }
    result
}

pub(super) fn trace_selection(
    config: ClampedRoutedConfig,
    tokens: usize,
    phase: ExecutionPhase,
    execution: ClampedMoeExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?phase,
        tokens,
        experts = config.experts,
        top_k = config.top_k,
        hidden_features = config.hidden,
        intermediate_features = config.intermediate,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected clamped CUDA MoE execution"
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
        assert!(compare(&reference, &close).equivalent);
        assert!(!compare(&reference, &drift).equivalent);
    }
}
