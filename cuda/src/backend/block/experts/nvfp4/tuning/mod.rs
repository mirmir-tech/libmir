use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, bf16};

use super::AutoNvFp4Experts;
use crate::{
    ExecutionPhase, MoeExecution, MoePlanRequest, PlanSource, Result,
    backend::tuning::MoeProfileExecution,
};

mod measure;

const ABSOLUTE_TOLERANCE: f32 = 0.5;
const RELATIVE_TOLERANCE: f32 = 0.01;

impl AutoNvFp4Experts {
    pub(super) fn tune(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let started = Instant::now();
        self.prepare_candidates();
        self.retain_compatible(input, selected, routing, output)?;
        let (winner, average, measured_elapsed) = self.measure(input, selected, routing, output)?;
        let execution = self.candidates[winner].execution;
        self.retain(winner);
        self.backend.auto_tuner().record_moe(
            self.profile,
            MoeProfileExecution::NvFp4(execution),
            average,
            started.elapsed().max(measured_elapsed),
        );
        trace_selection(self.request, execution, PlanSource::MeasuredStartup, Some(average));
        Ok(())
    }

    fn prepare_candidates(&mut self) {
        for &execution in candidate_executions(self.request) {
            if self.candidates.iter().any(|candidate| candidate.execution == execution) {
                continue;
            }
            match super::candidate::Candidate::new(
                &self.backend, self.request, self.activation, &self.weights, execution,
            ) {
                Ok(candidate) => self.candidates.push(candidate),
                Err(error) => tracing::debug!(
                    ?execution,
                    %error,
                    "discarded unavailable CUDA MoE tuning candidate"
                ),
            }
        }
    }

    fn retain_compatible(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.candidates[self.fallback].plan.execute(input, selected, routing, output)?;
        let context = self.backend.context().clone();
        let stream = self.backend.stream().clone();
        let reference = read(&context, &stream, output)?;
        let mut accepted = Vec::with_capacity(self.candidates.len());
        for (index, candidate) in self.candidates.iter_mut().enumerate() {
            let compatible = if index == self.fallback {
                true
            } else {
                candidate.plan.execute(input, selected, routing, output)?;
                equivalent(&reference, &read(&context, &stream, output)?)
            };
            if !compatible {
                tracing::warn!(
                    ?self.request,
                    execution = ?candidate.execution,
                    "rejected numerically incompatible CUDA MoE tuning candidate"
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
            .ok_or(crate::Error::InvalidExecutionPlan("MoE tuner rejected its fallback"))?;
        Ok(())
    }
}

fn candidate_executions(request: MoePlanRequest) -> &'static [MoeExecution] {
    match (request.phase, request.tokens) {
        (ExecutionPhase::Decode, 1) => &[MoeExecution::HybridW4A4, MoeExecution::IndexedGrouped],
        (ExecutionPhase::Prefill, _) => &[MoeExecution::Bucketed, MoeExecution::IndexedGrouped],
        (ExecutionPhase::Decode, _) => &[],
    }
}

fn read(
    context: &mircuda::Context,
    stream: &mircuda::Stream,
    output: &DeviceBuffer<bf16>,
) -> Result<Vec<bf16>> {
    let mut host = context.allocate_pinned::<bf16>(output.len())?;
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
    request: MoePlanRequest,
    execution: MoeExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        phase = ?request.phase,
        quantization = ?request.quantization,
        tokens = request.tokens,
        experts = request.experts,
        top_k = request.top_k,
        hidden_features = request.hidden_features,
        intermediate_features = request.intermediate_features,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA MoE execution"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_pre_capture_phases_have_candidates() {
        let decode = MoePlanRequest::nvfp4(ExecutionPhase::Decode, 1, 128, 8, 2_048, 768);
        let batch = MoePlanRequest { tokens: 4, ..decode };
        let prefill = MoePlanRequest {
            phase: ExecutionPhase::Prefill,
            tokens: 128,
            ..decode
        };

        assert_eq!(
            candidate_executions(decode),
            [MoeExecution::HybridW4A4, MoeExecution::IndexedGrouped]
        );
        assert!(candidate_executions(batch).is_empty());
        assert_eq!(
            candidate_executions(prefill),
            [MoeExecution::Bucketed, MoeExecution::IndexedGrouped]
        );
    }

    #[test]
    fn numerical_gate_accepts_bf16_rounding_only() {
        let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
        let close = [1.5, -20.5, 0.5].map(bf16::from_f32);
        let drift = [1.75, -20.0, 0.0].map(bf16::from_f32);

        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
