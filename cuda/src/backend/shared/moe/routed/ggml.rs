//! GGML routed experts of a shared-expert `MoE`: the router
//! and top-k as for the other storages, then the GGML expert kernels of the
//! dense routed path.

use mircuda::{DeviceBuffer, bf16};

use super::{super::AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, mxfp};
use crate::{
    CudaBackend, Result,
    backend::{
        dense::{GgmlExpertGeometry, GgmlExpertWeights, GgmlRoutedExperts, RoutedExpertsConfig},
        router::BoundRouter,
    },
};

pub(in super::super) struct GgmlRoutedExecution {
    router: BoundRouter,
    experts: GgmlRoutedExperts,
}

// The GGML expert kernels carry no `Debug`; the routing state names the plan.
impl std::fmt::Debug for GgmlRoutedExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgmlRoutedExecution")
            .field("router", &self.router)
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
        let router = mxfp::router(backend, config, weights, tokens)?;
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
        let selection = self.router.route(backend, input)?;
        self.experts
            .execute(input, selection.indices, selection.weights, weights, output)
    }
}
