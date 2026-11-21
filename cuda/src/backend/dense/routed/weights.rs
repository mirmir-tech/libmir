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
pub(in crate::backend) struct GgmlExpertWeights {
    pub(in crate::backend) gate: GgmlCheckpointWeight,
    pub(in crate::backend) up: GgmlCheckpointWeight,
    pub(in crate::backend) down: GgmlCheckpointWeight,
}

impl GgmlExpertWeights {
    /// Separate GGML gate, up and down banks, when all three are GGML.
    pub(in crate::backend) fn load(
        tensors: &CudaTensorSet,
        bindings: RoutedExpertBindings<'_>,
        (experts, hidden, intermediate): (usize, usize, usize),
    ) -> Result<Option<Self>> {
        let RoutedExpertBindings::SeparateGateUp { gate, up, down } = bindings else {
            return Ok(None);
        };
        if ![gate, up, down]
            .iter()
            .all(|binding| matches!(binding.storage, TensorStorage::Ggml { .. }))
        {
            return Ok(None);
        }
        let weights = Self {
            gate: GgmlCheckpointWeight::load_binding(tensors, gate)?,
            up: GgmlCheckpointWeight::load_binding(tensors, up)?,
            down: GgmlCheckpointWeight::load_binding(tensors, down)?,
        };
        weights.gate.validate(hidden, experts * intermediate)?;
        weights.up.validate(hidden, experts * intermediate)?;
        weights.down.validate(intermediate, experts * hidden)?;
        Ok(Some(weights))
    }
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
        let shape = (experts.count, hidden, config.intermediate_size);
        let banks = match GgmlExpertWeights::load(tensors, bindings.experts, shape)? {
            Some(weights) => RoutedExperts::Ggml(Box::new(weights)),
            None => RoutedExperts::Dense(Box::new(DenseExpertWeights::load(
                backend,
                tensors,
                bindings.experts,
                experts.count,
                hidden,
                config.intermediate_size,
            )?)),
        };
        Ok(Self { router, experts: banks })
    }
}
