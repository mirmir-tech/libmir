use mircuda::{DeviceBuffer, bf16};

use super::{ClampedRoutedConfig, weights::ClampedRoutedExpertWeights};
use crate::{
    CudaBackend, ExecutionPhase, PlanSource, Result,
    backend::tuning::{
        ClampedMoeExecution, ClampedMoeStorage, MoeProfileExecution, MoeProfileRequest,
    },
    kernels::ClampedRoutedKernels,
};

mod candidate;
mod marlin;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod tuning;

use candidate::Candidate;

pub(super) struct AutoClampedExperts {
    backend: CudaBackend,
    config: ClampedRoutedConfig,
    tokens: usize,
    phase: ExecutionPhase,
    profile: MoeProfileRequest,
    kernels: ClampedRoutedKernels,
    candidates: Vec<Candidate>,
    fallback: usize,
    tunable: bool,
}

impl AutoClampedExperts {
    pub(super) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        tokens: usize,
        phase: ExecutionPhase,
        weights: &ClampedRoutedExpertWeights,
        kernels: ClampedRoutedKernels,
    ) -> Option<Self> {
        let storage = match weights {
            ClampedRoutedExpertWeights::Native(_) => ClampedMoeStorage::Native,
            ClampedRoutedExpertWeights::Mlx(_) => ClampedMoeStorage::Mlx,
            ClampedRoutedExpertWeights::Dense(_) => return None,
        };
        let profile = MoeProfileRequest::clamped(
            phase,
            tokens,
            config.experts,
            config.top_k,
            config.hidden,
            config.intermediate,
            storage,
        );
        let cached =
            backend
                .auto_tuner()
                .lookup_clamped_moe(profile)
                .and_then(|(execution, source)| match execution {
                    MoeProfileExecution::Clamped(execution) => Some((execution, source)),
                    MoeProfileExecution::NvFp4(_)
                    | MoeProfileExecution::Affine(_)
                    | MoeProfileExecution::MxFp4(_)
                    | MoeProfileExecution::MxFp8(_) => None,
                });
        let mut selected = cached.map_or(ClampedMoeExecution::FusedReduce, |value| {
            value.0.for_batch(tokens, config.experts, config.top_k)
        });
        let candidate = match candidate(backend, config, tokens, weights, &kernels, selected) {
            Ok(candidate) => candidate,
            Err(error) => {
                tracing::warn!(%error, ?selected, "cached clamped CUDA MoE plan is unavailable");
                selected = ClampedMoeExecution::FusedReduce;
                Candidate::portable(kernels.clone(), selected)
            },
        };
        let tunable = cached.is_none()
            && backend.auto_tuner().prepares_candidates(PlanSource::Heuristic)
            && (phase == ExecutionPhase::Prefill || tokens == 1);
        if let Some((_, source)) = cached {
            tuning::trace_selection(config, tokens, phase, selected, source, None);
        }
        Some(Self {
            backend: backend.clone(),
            config,
            tokens,
            phase,
            profile,
            kernels,
            candidates: vec![candidate],
            fallback: 0,
            tunable,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        weights: &ClampedRoutedExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        activated: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.tunable {
            self.select(weights, input, selected, routing, activated, partial, output);
        }
        self.candidates[self.fallback].execute(
            self.backend.stream(),
            weights,
            input,
            selected,
            routing,
            activated,
            partial,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn select(
        &mut self,
        weights: &ClampedRoutedExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        activated: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) {
        self.tunable = false;
        let cached = self.backend.auto_tuner().lookup_clamped_moe(self.profile).and_then(
            |(execution, source)| match execution {
                MoeProfileExecution::Clamped(execution) => Some((execution, source)),
                MoeProfileExecution::NvFp4(_)
                | MoeProfileExecution::Affine(_)
                | MoeProfileExecution::MxFp4(_)
                | MoeProfileExecution::MxFp8(_) => None,
            },
        );
        if let Some((execution, source)) = cached {
            let execution =
                execution.for_batch(self.tokens, self.config.experts, self.config.top_k);
            if let Err(error) = self.retain_execution(weights, execution) {
                tracing::warn!(%error, ?execution, "cached clamped CUDA MoE plan is unavailable");
            }
            tuning::trace_selection(self.config, self.tokens, self.phase, execution, source, None);
            return;
        }
        if !self.backend.auto_tuner().claim_moe(self.profile) {
            return;
        }
        if let Err(error) = self.tune(weights, input, selected, routing, activated, partial, output)
        {
            self.backend.auto_tuner().abandon_moe(self.profile);
            self.retain(self.fallback);
            tracing::warn!(%error, "clamped CUDA MoE tuning failed; retaining fused fallback");
        }
    }

    fn retain_execution(
        &mut self,
        weights: &ClampedRoutedExpertWeights,
        execution: ClampedMoeExecution,
    ) -> Result<()> {
        let index = self.candidates.iter().position(|candidate| candidate.execution == execution);
        let index = if let Some(index) = index {
            index
        } else {
            self.candidates.push(candidate(
                &self.backend, self.config, self.tokens, weights, &self.kernels, execution,
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

fn candidate(
    backend: &CudaBackend,
    config: ClampedRoutedConfig,
    tokens: usize,
    weights: &ClampedRoutedExpertWeights,
    kernels: &ClampedRoutedKernels,
    execution: ClampedMoeExecution,
) -> Result<Candidate> {
    match execution {
        ClampedMoeExecution::FusedReduce | ClampedMoeExecution::RouteParallel => {
            Ok(Candidate::portable(kernels.clone(), execution))
        },
        ClampedMoeExecution::MarlinN128K128
        | ClampedMoeExecution::MarlinN128K64
        | ClampedMoeExecution::MarlinN64K128
        | ClampedMoeExecution::MarlinM64N256K64
        | ClampedMoeExecution::MarlinM64N128K64
        | ClampedMoeExecution::MarlinM64N64K128 => {
            let ClampedRoutedExpertWeights::Native(weights) = weights else {
                return Err(crate::Error::InvalidExecutionPlan(
                    "MXFP4 Marlin requires native clamped expert weights",
                ));
            };
            Candidate::marlin(backend, config, tokens, weights, execution)
        },
    }
}
