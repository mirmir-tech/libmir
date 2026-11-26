mod ggml;

pub(in crate::backend) use ggml::{GgmlExpertBank, GgmlExpertWeights};
use mircuda::DeviceBuffer;
use models::weights::{RoutedFeedForwardBindings, SharedRoutedFeedForwardBindings, TensorBinding};

use super::super::DenseSwiGluConfig;
use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    backend::{CheckpointProjectionWeight, DenseExpertWeights},
};

/// A layer's router and expert banks, bound at execution like the gated
/// MLP's projections so one executor serves every layer.
#[derive(Clone, Debug)]
pub struct RoutedExpertWeights {
    pub(super) router: CudaTensor,
    pub(super) experts: RoutedExperts,
    /// The `[experts]` F32 selection-only bias of a sigmoid router.
    pub(super) correction_bias: Option<DeviceBuffer<f32>>,
    pub(super) shared: Option<SharedExpertWeights>,
}

/// The gated MLP of a shared expert (GLM-4.5).
#[derive(Clone, Debug)]
pub(in super::super) struct SharedExpertWeights {
    pub(in super::super) gate: CheckpointProjectionWeight,
    pub(in super::super) up: CheckpointProjectionWeight,
    pub(in super::super) down: CheckpointProjectionWeight,
}

#[derive(Clone, Debug)]
pub(in super::super) enum RoutedExperts {
    Dense(Box<DenseExpertWeights>),
    Ggml(Box<GgmlExpertWeights>),
}

impl RoutedExpertWeights {
    /// Multiplies the down products of the GGML banks and of a GGML shared
    /// expert by `factor`, through their output scales. Dense down weights
    /// were scaled in place at upload (`residual::fold_dense`), which refuses
    /// every other storage, so they take nothing here.
    pub(crate) fn scale_ggml_down(&mut self, backend: &CudaBackend, factor: f32) -> Result<()> {
        if let RoutedExperts::Ggml(weights) = &mut self.experts {
            weights.scale_down(backend, factor)?;
        }
        if let Some(SharedExpertWeights {
            down: CheckpointProjectionWeight::Ggml(down),
            ..
        }) = &mut self.shared
        {
            *down = down.scaled(backend, factor)?;
        }
        Ok(())
    }

    pub(crate) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        bindings: RoutedFeedForwardBindings<'_>,
        config: DenseSwiGluConfig,
    ) -> Result<Self> {
        let experts = config
            .experts
            .ok_or(Error::InvalidExecutionPlan("dense layer config declares no routed experts"))?
            .experts;
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
        Ok(Self {
            router,
            experts: banks,
            correction_bias: None,
            shared: None,
        })
    }

    /// Routed experts with a shared expert added ungated, and the sigmoid
    /// router's correction bias when it has one.
    pub(crate) fn load_shared(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        bindings: SharedRoutedFeedForwardBindings<'_>,
        config: DenseSwiGluConfig,
    ) -> Result<Self> {
        if bindings.shared_output_gate.is_some() {
            return Err(Error::UnsupportedDecoderLayer(
                "dense CUDA has no gated shared expert".into(),
            ));
        }
        let routed = RoutedFeedForwardBindings {
            router: bindings.router,
            experts: bindings.routed,
        };
        let mut weights = Self::load(backend, tensors, routed, config)?;
        weights.correction_bias = bindings
            .router_correction_bias
            .map(|binding| {
                tensors.get(&binding.source).and_then(CudaTensor::as_f32).cloned().ok_or_else(
                    || Error::DTypeMismatch {
                        name: binding.source.clone(),
                        expected: "F32",
                    },
                )
            })
            .transpose()?;
        let projection =
            |binding: &TensorBinding| CheckpointProjectionWeight::load_binding(tensors, binding);
        weights.shared = Some(SharedExpertWeights {
            gate: projection(bindings.shared_gate)?,
            up: projection(bindings.shared_up)?,
            down: projection(bindings.shared_down)?,
        });
        Ok(weights)
    }
}
