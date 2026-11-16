use models::weights::{RoutedExpertBindings, RoutedFeedForwardBindings, TensorStorage};

use super::super::DenseSwiGluConfig;
use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, GgmlCheckpointWeight, Result,
    backend::DenseExpertWeights,
};

/// A layer's router and expert banks, bound at execution like the gated
/// MLP's projections so one executor serves every layer.
#[derive(Clone, Debug)]
pub struct RoutedExpertWeights {
    pub(super) router: CudaTensor,
    pub(super) experts: RoutedExperts,
}

#[derive(Clone, Debug)]
pub(super) enum RoutedExperts {
    Dense(Box<DenseExpertWeights>),
    Ggml(Box<GgmlExpertWeights>),
}

/// Stacked GGML banks `[experts, out, in]`, each bound as one matrix of
/// `experts * out` rows.
#[derive(Clone, Debug)]
pub(super) struct GgmlExpertWeights {
    pub(super) gate: GgmlCheckpointWeight,
    pub(super) up: GgmlCheckpointWeight,
    pub(super) down: GgmlCheckpointWeight,
}

impl RoutedExpertWeights {
    pub(crate) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        bindings: RoutedFeedForwardBindings<'_>,
        config: DenseSwiGluConfig,
    ) -> Result<Self> {
        let experts = config
            .experts
            .ok_or(Error::InvalidExecutionPlan("dense layer config declares no routed experts"))?;
        let hidden = config.attention.hidden_size;
        let router = tensors
            .get(&bindings.router.source)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(bindings.router.source.clone()))?;
        if router.dtype() != CudaTensorDType::Bf16 || router.shape() != [experts.count, hidden] {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA routed experts need a dense [{}, {hidden}] router: {}",
                experts.count, bindings.router.source
            )));
        }
        let banks = match bindings.experts {
            RoutedExpertBindings::SeparateGateUp { gate, up, down }
                if [gate, up, down]
                    .iter()
                    .all(|binding| matches!(binding.storage, TensorStorage::Ggml { .. })) =>
            {
                let weights = GgmlExpertWeights {
                    gate: GgmlCheckpointWeight::load_binding(tensors, gate)?,
                    up: GgmlCheckpointWeight::load_binding(tensors, up)?,
                    down: GgmlCheckpointWeight::load_binding(tensors, down)?,
                };
                let intermediate = config.intermediate_size;
                weights.gate.validate(hidden, experts.count * intermediate)?;
                weights.up.validate(hidden, experts.count * intermediate)?;
                weights.down.validate(intermediate, experts.count * hidden)?;
                RoutedExperts::Ggml(Box::new(weights))
            },
            other => RoutedExperts::Dense(Box::new(DenseExpertWeights::load(
                backend,
                tensors,
                other,
                experts.count,
                hidden,
                config.intermediate_size,
            )?)),
        };
        Ok(Self { router, experts: banks })
    }
}
