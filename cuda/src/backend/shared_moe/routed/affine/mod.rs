use mircuda::{DeviceBuffer, bf16};

use self::candidate::Candidate;
use super::super::{AffineSharedExpertMoeConfig, weights::AffineRoutedMoeWeights};
use crate::{
    CudaBackend, ExecutionPhase, PlanSource, Result,
    backend::tuning::{AffineMoeExecution, MoeProfileExecution, MoeProfileRequest},
};

mod candidate;
mod tuning;

#[derive(Debug)]
pub(in crate::backend::shared_moe) struct AutoAffineRoutedExperts {
    backend: CudaBackend,
    config: AffineSharedExpertMoeConfig,
    tokens: usize,
    profile: MoeProfileRequest,
    candidates: Vec<Candidate>,
    fallback: usize,
    tunable: bool,
}

impl AutoAffineRoutedExperts {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        _weights: &AffineRoutedMoeWeights,
        tokens: usize,
    ) -> Result<Self> {
        let phase = if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let profile = MoeProfileRequest::affine(
            phase,
            tokens,
            config.expert_count,
            config.top_k,
            config.hidden_size,
            config.routed_intermediate_size,
            config.group_size,
            config.expert_bits,
            config.activation,
        );
        let cached =
            backend.auto_tuner().lookup_moe(profile).and_then(
                |(execution, source)| match execution {
                    MoeProfileExecution::Affine(execution) => Some((execution, source)),
                    MoeProfileExecution::NvFp4(_)
                    | MoeProfileExecution::Clamped(_)
                    | MoeProfileExecution::MxFp4(_)
                    | MoeProfileExecution::MxFp8(_) => None,
                },
            );
        let execution = cached.map_or(AffineMoeExecution::FusedGated, |value| value.0);
        let (candidate, cache_applied) = match Candidate::new(backend, config, tokens, execution) {
            Ok(candidate) => (candidate, cached.is_some()),
            Err(error) if execution != AffineMoeExecution::FusedGated => {
                tracing::warn!(
                    ?execution,
                    %error,
                    "cached affine MoE candidate is unavailable; using fused fallback"
                );
                (Candidate::new(backend, config, tokens, AffineMoeExecution::FusedGated)?, false)
            },
            Err(error) => return Err(error),
        };
        if let Some((execution, source)) = cached.filter(|_| cache_applied) {
            tuning::trace_selection(profile, execution, source, None);
        }
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            profile,
            candidates: vec![candidate],
            fallback: 0,
            tunable: cached.is_none()
                && backend.auto_tuner().prepares_candidates(PlanSource::Heuristic),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &AffineRoutedMoeWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.tunable {
            self.select(input, selected, routing, weights, intermediate, output);
        }
        self.candidates[self.fallback]
            .execute(input, selected, routing, weights, intermediate, output)
    }

    #[allow(clippy::too_many_arguments)]
    fn select(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &AffineRoutedMoeWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) {
        self.tunable = false;
        let cached =
            self.backend
                .auto_tuner()
                .lookup_moe(self.profile)
                .and_then(|(execution, source)| match execution {
                    MoeProfileExecution::Affine(execution) => Some((execution, source)),
                    MoeProfileExecution::NvFp4(_)
                    | MoeProfileExecution::Clamped(_)
                    | MoeProfileExecution::MxFp4(_)
                    | MoeProfileExecution::MxFp8(_) => None,
                });
        if let Some((execution, source)) = cached {
            if let Err(error) = self.retain_execution(execution) {
                tracing::warn!(?execution, %error, "cached affine MoE candidate is unavailable");
            } else {
                tuning::trace_selection(self.profile, execution, source, None);
            }
            return;
        }
        if !self.backend.auto_tuner().claim_moe(self.profile) {
            return;
        }
        if let Err(error) = self.tune(input, selected, routing, weights, intermediate, output) {
            self.backend.auto_tuner().abandon_moe(self.profile);
            self.retain(self.fallback);
            tracing::warn!(%error, "CUDA affine MoE tuning failed; retaining fused fallback");
        }
    }

    fn retain_execution(&mut self, execution: AffineMoeExecution) -> Result<()> {
        let index = self.candidates.iter().position(|candidate| candidate.execution == execution);
        let index = if let Some(index) = index {
            index
        } else {
            self.candidates
                .push(Candidate::new(&self.backend, self.config, self.tokens, execution)?);
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
