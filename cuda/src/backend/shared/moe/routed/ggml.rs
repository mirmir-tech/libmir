//! GGML routed experts of a shared-expert `MoE`: the router
//! and top-k as for the other storages, then the GGML expert kernels of the
//! dense routed path.

use mircuda::{DeviceBuffer, bf16};

use super::{super::AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, mxfp};
use crate::{
    CudaBackend, Result,
    backend::{
        dense::{GgmlExpertGeometry, GgmlExpertWeights, GgmlRoutedExperts, RoutedExpertsConfig},
        linear::CheckpointProjection,
    },
    kernels::RouterUnitTopK,
};

pub(in super::super) struct GgmlRoutedExecution {
    router: Box<CheckpointProjection>,
    top_k: RouterUnitTopK,
    scores: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    routing: DeviceBuffer<bf16>,
    experts: GgmlRoutedExperts,
}

// The GGML expert kernels carry no `Debug`; the routing state names the plan.
impl std::fmt::Debug for GgmlRoutedExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgmlRoutedExecution")
            .field("router", &self.router)
            .field("top_k", &self.top_k)
            .finish_non_exhaustive()
    }
}

impl GgmlRoutedExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: &AffineSharedExpertMoeWeights,
        tokens: usize,
    ) -> Result<Self> {
        let (router, top_k, scores, selected, routing) =
            mxfp::routing(backend, config, weights, tokens, 0)?;
        let geometry = GgmlExpertGeometry {
            hidden: config.hidden_size,
            intermediate: config.routed_intermediate_size,
            experts: RoutedExpertsConfig {
                count: config.expert_count,
                top_k: config.top_k,
            },
            activation: config.activation,
        };
        Ok(Self {
            router,
            top_k,
            scores,
            selected,
            routing,
            experts: GgmlRoutedExperts::new(backend, geometry, tokens)?,
        })
    }

    pub(super) fn execute(
        &mut self,
        backend: &CudaBackend,
        input: &DeviceBuffer<bf16>,
        weights: &GgmlExpertWeights,
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
