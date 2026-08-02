use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, bf16};

use super::{
    super::{ClampedRoutedConfig, weights::ClampedRoutedExpertWeights},
    AutoClampedExperts,
};
use crate::{
    ExecutionPhase, PlanSource, Result,
    backend::{
        clamped_routed::experts::candidate::Candidate,
        tuning::{ClampedMoeExecution, MoeProfileExecution},
    },
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
        self.prepare_candidates();
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

    fn prepare_candidates(&mut self) {
        for execution in [ClampedMoeExecution::FusedReduce, ClampedMoeExecution::RouteParallel] {
            if self.candidates.iter().any(|candidate| candidate.execution == execution) {
                continue;
            }
            self.candidates.push(Candidate::new(self.kernels.clone(), execution));
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
                equivalent(&reference, &read(&self.backend, output)?)
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
        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
