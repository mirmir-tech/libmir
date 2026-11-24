use mircuda::{DeviceBuffer, bf16};

use super::mxfp;
use crate::{
    CudaBackend, ExecutionPhase, Result,
    backend::{
        block::experts::{ExpertWeights, Experts},
        router::BoundRouter,
        shared::moe::{AffineSharedExpertMoeConfig, weights::AffineSharedExpertMoeWeights},
    },
};

#[derive(Debug)]
pub(in crate::backend::shared::moe) struct NvFp4RoutedExecution {
    router: BoundRouter,
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
        Ok(Self {
            router: mxfp::router(backend, config, weights, tokens)?,
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
        self.router.route(backend, input).map(|_| ())
    }

    pub(super) fn execute_prepared(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &ExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let selection = self.router.selection();
        self.experts
            .execute(input, selection.indices, selection.weights, weights, output)
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
            packed,
            scales,
            self.router.selection().indices,
            self.router.selection().weights,
            weights,
            residual,
            shared,
            output,
        )
    }
}
