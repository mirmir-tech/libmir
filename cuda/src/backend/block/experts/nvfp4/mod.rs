use mircuda::{DeviceBuffer, bf16};
use models::weights::BlockActivationMode;

use self::candidate::Candidate;
use crate::{
    CudaBackend, ExecutionPhase, GatedActivation, MoeExecution, MoePlanRequest, NvFp4ExpertBank,
    PlanSource, Result,
    backend::tuning::{MoeProfileExecution, MoeProfileRequest},
};

mod candidate;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod tuning;

#[derive(Debug)]
pub(in crate::backend) struct AutoNvFp4Experts {
    backend: CudaBackend,
    request: MoePlanRequest,
    profile: MoeProfileRequest,
    activation: GatedActivation,
    weights: [NvFp4ExpertBank; 3],
    candidates: Vec<Candidate>,
    fallback: usize,
    tunable: bool,
    weight_only: bool,
}

impl AutoNvFp4Experts {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: &CudaBackend,
        phase: ExecutionPhase,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        activation_mode: BlockActivationMode,
    ) -> Result<Self> {
        let config = gate.config();
        let request = MoePlanRequest::nvfp4(
            phase,
            tokens,
            config.experts,
            selected,
            config.input_features,
            config.output_features,
        );
        let planned = backend.execution_planner().plan_moe(request)?;
        let weight_only = activation_mode == BlockActivationMode::WeightOnly;
        let profile = MoeProfileRequest::nvfp4(request, activation, weight_only);
        let cached = (planned.source() != PlanSource::ExplicitPolicy)
            .then(|| backend.auto_tuner().lookup_moe(profile))
            .flatten()
            .and_then(|(execution, source)| match execution {
                MoeProfileExecution::NvFp4(execution) => Some((execution, source)),
                MoeProfileExecution::Affine(_)
                | MoeProfileExecution::Clamped(_)
                | MoeProfileExecution::MxFp4(_)
                | MoeProfileExecution::MxFp8(_) => None,
            });
        let weights = [gate, up, down];
        let fallback_execution = fallback_execution(phase, weight_only, planned.execution());
        let selected_execution = cached.map_or(fallback_execution, |value| value.0);
        let (candidate, cache_applied) =
            match Candidate::new(backend, request, activation, &weights, selected_execution) {
                Ok(candidate) => (candidate, cached.is_some()),
                Err(error) if selected_execution != fallback_execution => {
                    tracing::warn!(
                        ?selected_execution,
                        %error,
                        "cached CUDA MoE candidate is unavailable; using planner fallback"
                    );
                    (
                        Candidate::new(backend, request, activation, &weights, fallback_execution)?,
                        false,
                    )
                },
                Err(error) => return Err(error),
            };
        let tunable_phase = if weight_only {
            true
        } else {
            phase == ExecutionPhase::Prefill || tokens == 1
        };
        let tunable = cached.is_none()
            && backend.auto_tuner().prepares_moe_candidates(profile, planned.source())
            && tunable_phase;
        if let Some((execution, source)) = cached.filter(|_| cache_applied) {
            tuning::trace_selection(request, execution, source, None);
        }
        Ok(Self {
            backend: backend.clone(),
            request,
            profile,
            activation,
            weights,
            candidates: vec![candidate],
            fallback: 0,
            tunable,
            weight_only,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.tunable {
            self.select(input, selected, routing, output);
        }
        self.candidates[self.fallback].plan.execute(input, selected, routing, output)
    }

    pub(super) fn prequant_scale(&self) -> Option<DeviceBuffer<f32>> {
        (!self.tunable)
            .then(|| self.candidates[self.fallback].plan.prequant_scale())
            .flatten()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_prequantized_residual_shared(
        &mut self,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        residual: &DeviceBuffer<bf16>,
        shared: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.candidates[self.fallback].plan.execute_prequantized_residual_shared(
            packed, scales, selected, routing, residual, shared, output,
        )
    }

    fn select(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) {
        self.tunable = false;
        let cached =
            self.backend
                .auto_tuner()
                .lookup_moe(self.profile)
                .and_then(|(execution, source)| match execution {
                    MoeProfileExecution::NvFp4(execution) => Some((execution, source)),
                    MoeProfileExecution::Affine(_)
                    | MoeProfileExecution::Clamped(_)
                    | MoeProfileExecution::MxFp4(_)
                    | MoeProfileExecution::MxFp8(_) => None,
                });
        if let Some((execution, source)) = cached {
            if let Err(error) = self.retain_execution(execution) {
                tracing::warn!(?execution, %error, "cached CUDA MoE candidate became unavailable");
            } else {
                tuning::trace_selection(self.request, execution, source, None);
            }
            return;
        }
        if !self.backend.auto_tuner().claim_moe(self.profile) {
            return;
        }
        if let Err(error) = self.tune(input, selected, routing, output) {
            self.backend.auto_tuner().abandon_moe(self.profile);
            self.retain(self.fallback);
            tracing::warn!(
                ?self.request,
                %error,
                "CUDA MoE tuning failed; retaining the stable fallback"
            );
        }
    }

    fn retain_execution(&mut self, execution: MoeExecution) -> Result<()> {
        let index = self.candidates.iter().position(|candidate| candidate.execution == execution);
        let index = if let Some(index) = index {
            index
        } else {
            self.candidates.push(Candidate::new(
                &self.backend, self.request, self.activation, &self.weights, execution,
            )?);
            self.candidates.len() - 1
        };
        self.retain(index);
        Ok(())
    }

    fn retain(&mut self, selected: usize) {
        let selected = self.candidates.swap_remove(selected);
        self.candidates.clear();
        self.candidates.push(selected);
        self.fallback = 0;
    }
}

const fn fallback_execution(
    phase: ExecutionPhase,
    weight_only: bool,
    planned: MoeExecution,
) -> MoeExecution {
    if weight_only && matches!(phase, ExecutionPhase::Decode) {
        MoeExecution::SelectedWeightOnly
    } else {
        planned
    }
}
