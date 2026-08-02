use std::time::Duration;

use mircuda::{Context, DeviceBuffer, Stream, bf16};
use runtime::tuning::select_fastest_candidate;

use self::candidate::{Candidate, candidates, measure};
use super::CudaBackend;
use crate::{DenseExecution, DensePlanRequest, ExecutionPhase, PlanSource, Result};

mod candidate;
mod validation;

#[derive(Debug)]
pub(in crate::backend) struct AutoBf16Plan {
    request: DensePlanRequest,
    candidates: Vec<Candidate>,
    selected: Option<usize>,
    fallback: usize,
    tunable: bool,
    context: Context,
    stream: Stream,
    tuner: crate::backend::tuning::CudaAutoTuner,
}

impl AutoBf16Plan {
    pub(super) fn new(backend: &CudaBackend, request: DensePlanRequest) -> Result<Self> {
        let planned = backend.execution_planner().plan_dense(request)?;
        let cached = (planned.source() != PlanSource::ExplicitPolicy)
            .then(|| backend.inner.tuner.lookup_dense(request))
            .flatten();
        let prepare_candidates = cached.is_none()
            && backend.inner.tuner.prepares_candidates(planned.source())
            && (request.phase == ExecutionPhase::Prefill || request.tokens == 1);
        let cached_execution = cached.map(|(execution, _)| execution);
        let executions = if planned.source() == PlanSource::ExplicitPolicy {
            vec![planned.execution()]
        } else {
            candidate::initial_executions(planned.execution(), cached_execution, request.phase)
        };
        let mut prepared = Vec::with_capacity(1);
        let execution_count = executions.len();
        for (index, execution) in executions.into_iter().enumerate() {
            match Candidate::new(backend, request, execution) {
                Ok(candidate) => {
                    prepared.push(candidate);
                    break;
                },
                Err(error) if index + 1 < execution_count => {
                    tracing::debug!(
                        ?execution,
                        %error,
                        "discarded unavailable CUDA dense tuning candidate"
                    );
                },
                Err(error) => return Err(error),
            }
        }
        Ok(Self {
            request,
            selected: (!prepare_candidates).then_some(0),
            candidates: prepared,
            fallback: 0,
            tunable: prepare_candidates,
            context: backend.inner.context.clone(),
            stream: backend.inner.stream.clone(),
            tuner: backend.inner.tuner.clone(),
        })
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.selected.is_none() {
            self.select(input, weight, output)?;
        }
        self.candidates[self.selected.unwrap_or(self.fallback)]
            .plan
            .execute(&self.stream, input, weight, output)
    }

    pub(super) const fn request(&self) -> DensePlanRequest {
        self.request
    }

    fn select(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if let Some((execution, source)) = self.tuner.lookup_dense(self.request) {
            let index = if let Some(index) =
                self.candidates.iter().position(|candidate| candidate.execution == execution)
            {
                index
            } else {
                self.candidates.push(Candidate::new_with_resources(
                    &self.context, &self.stream, self.request, execution,
                )?);
                self.candidates.len() - 1
            };
            self.retain(index);
            trace_selection(self.request, execution, source, None);
            return Ok(());
        }
        if !self.tunable {
            return Ok(());
        }
        if !self.tuner.claim_dense(self.request) {
            return Ok(());
        }
        for execution in candidates(self.request) {
            if !self.candidates.iter().any(|candidate| candidate.execution == execution) {
                match Candidate::new_with_resources(
                    &self.context, &self.stream, self.request, execution,
                ) {
                    Ok(candidate) => self.candidates.push(candidate),
                    Err(error) => {
                        tracing::debug!(
                            ?execution,
                            %error,
                            "discarded unavailable CUDA dense tuning candidate"
                        );
                    },
                }
            }
        }
        self.fallback = validation::retain_compatible(
            &self.context,
            &self.stream,
            self.request,
            &mut self.candidates,
            self.fallback,
            input,
            weight,
            output,
        )?;
        match self.measure(input, weight, output) {
            Ok((selected, average, elapsed)) => {
                let execution = self.candidates[selected].execution;
                self.retain(selected);
                self.tuner.record_dense(self.request, execution, average, elapsed);
                trace_selection(
                    self.request,
                    execution,
                    PlanSource::MeasuredStartup,
                    Some(average),
                );
                Ok(())
            },
            Err(error) => {
                self.tuner.abandon_dense(self.request);
                tracing::warn!(
                    ?error,
                    ?self.request,
                    "CUDA dense tuning failed; retaining the stable fallback"
                );
                self.retain(self.fallback);
                Ok(())
            },
        }
    }

    fn retain(&mut self, selected: usize) {
        let selected = self.candidates.swap_remove(selected);
        self.candidates.clear();
        self.candidates.push(selected);
        self.selected = Some(0);
        self.fallback = 0;
    }

    fn measure(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<(usize, Duration, Duration)> {
        let (warmup, iterations) = self.tuner.iterations(self.request.tokens);
        let mut timings = Vec::with_capacity(self.candidates.len());
        let mut elapsed = Duration::ZERO;
        for candidate in &mut self.candidates {
            for _ in 0..warmup {
                candidate.plan.execute(&self.stream, input, weight, output)?;
            }
            let average = measure(
                &self.context,
                &self.stream,
                &mut candidate.plan,
                input,
                weight,
                output,
                iterations,
            )?;
            elapsed =
                elapsed.saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
            timings.push(average);
        }
        let fastest = timings
            .iter()
            .enumerate()
            .min_by_key(|(_, duration)| **duration)
            .map(|value| value.0)
            .ok_or(crate::Error::InvalidExecutionPlan("dense tuner has no candidates"))?;
        let selected = select_fastest_candidate(
            fastest,
            self.fallback,
            &timings,
            self.tuner.minimum_improvement_bps(),
        );
        Ok((selected, timings[selected], elapsed))
    }
}

fn trace_selection(
    request: DensePlanRequest,
    execution: DenseExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        phase = ?request.phase,
        role = ?request.role,
        tokens = request.tokens,
        input_features = request.input_features,
        output_features = request.output_features,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA dense execution profile"
    );
}
