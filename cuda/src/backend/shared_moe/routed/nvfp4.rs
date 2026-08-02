use mircuda::{DeviceBuffer, bf16};

use super::mxfp;
use crate::{
    CudaBackend, ExecutionPhase, Result,
    backend::{
        block::experts::{ExpertWeights, Experts},
        linear::CheckpointProjection,
        shared_moe::{AffineSharedExpertMoeConfig, weights::AffineSharedExpertMoeWeights},
    },
    kernels::RouterUnitTopK,
};

#[derive(Debug)]
pub(in crate::backend::shared_moe) struct NvFp4RoutedExecution {
    router: Box<CheckpointProjection>,
    top_k: RouterUnitTopK,
    scores: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    routing: DeviceBuffer<bf16>,
    experts: Box<Experts>,
}

impl NvFp4RoutedExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: &AffineSharedExpertMoeWeights,
        expert_weights: &ExpertWeights,
        tokens: usize,
    ) -> Result<Self> {
        let (router, top_k, scores, selected, routing) =
            mxfp::routing(backend, config, weights, tokens, 4)?;
        Ok(Self {
            router,
            top_k,
            scores,
            selected,
            routing,
            experts: Box::new(Experts::new(
                backend,
                if tokens == 1 {
                    ExecutionPhase::Decode
                } else {
                    ExecutionPhase::Prefill
                },
                tokens,
                config.top_k,
                config.activation,
                expert_weights,
            )?),
        })
    }

    pub(super) fn execute(
        &mut self,
        backend: &CudaBackend,
        input: &DeviceBuffer<bf16>,
        weights: &ExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.router.execute(input, &mut self.scores)?;
        self.top_k.execute(
            backend.stream(),
            &self.scores,
            &mut self.selected,
            &mut self.routing,
        )?;
        self.experts.execute(input, &self.selected, &self.routing, weights, output)
    }
}
