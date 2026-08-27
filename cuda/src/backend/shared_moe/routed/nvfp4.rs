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
        phase: ExecutionPhase,
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
                phase,
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
        self.prepare_routing(backend, input)?;
        self.execute_prepared(input, weights, output)
    }

    pub(super) fn prepare_routing(
        &mut self,
        backend: &CudaBackend,
        input: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.router.execute(input, &mut self.scores)?;
        self.top_k
            .execute(backend.stream(), &self.scores, &mut self.selected, &mut self.routing)
    }

    pub(super) fn execute_prepared(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &ExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.experts.execute(input, &self.selected, &self.routing, weights, output)
    }

    pub(super) fn prequant_scale(&self) -> Option<DeviceBuffer<f32>> {
        self.experts.nvfp4_prequant_scale()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_prequantized_residual_shared(
        &mut self,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        weights: &ExpertWeights,
        residual: &DeviceBuffer<bf16>,
        shared: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.experts.execute_nvfp4_prequantized_residual_shared(
            packed, scales, &self.selected, &self.routing, weights, residual, shared, output,
        )
    }
}
