use mircuda::{DeviceBuffer, bf16};
use models::weights::BlockActivationMode;

use self::nvfp4::AutoNvFp4Experts;
use crate::{
    CudaBackend, Error, ExecutionPhase, GatedActivation, NvFp4ExpertBank, Result,
    backend::{
        dense::{GgmlExpertWeights, GgmlRoutedExperts, RoutedExpertsConfig},
        linear::{DenseExpertWeights, SelectedDenseMoeBf16},
    },
};

mod nvfp4;

#[derive(Clone, Debug)]
pub(in crate::backend) enum ExpertWeights {
    NvFp4 {
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        activation_mode: BlockActivationMode,
    },
    Dense(DenseExpertWeights),
    /// GGML banks read in place from a GGUF file.
    Ggml(Box<GgmlExpertWeights>),
}

#[derive(Debug)]
pub(in crate::backend) enum Experts {
    NvFp4(Box<AutoNvFp4Experts>),
    Dense {
        operation: Box<SelectedDenseMoeBf16>,
        intermediate: DeviceBuffer<bf16>,
    },
    Ggml(Box<GgmlRoutedExperts>),
}

impl Experts {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        phase: ExecutionPhase,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        weights: &ExpertWeights,
    ) -> Result<Self> {
        match weights {
            ExpertWeights::Dense(weights) => Ok(Self::Dense {
                operation: Box::new(SelectedDenseMoeBf16::new(
                    backend,
                    tokens,
                    selected,
                    weights,
                    activation.into(),
                )?),
                intermediate: backend
                    .pool()
                    .allocate(backend.stream(), weights.intermediate_elements(tokens, selected)?)?,
            }),
            ExpertWeights::Ggml(weights) => {
                let experts = RoutedExpertsConfig {
                    count: weights.experts(),
                    top_k: selected,
                };
                let geometry = weights.geometry(experts, activation);
                Ok(Self::Ggml(Box::new(GgmlRoutedExperts::new(backend, geometry, tokens)?)))
            },
            ExpertWeights::NvFp4 { gate, up, down, activation_mode } => AutoNvFp4Experts::new(
                backend,
                phase,
                tokens,
                selected,
                activation,
                gate.clone(),
                up.clone(),
                down.clone(),
                *activation_mode,
            )
            .map(Box::new)
            .map(Self::NvFp4),
        }
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &ExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weights) {
            (Self::NvFp4(experts), ExpertWeights::NvFp4 { .. }) => {
                experts.execute(input, selected, routing, output)
            },
            (Self::Dense { operation, intermediate }, ExpertWeights::Dense(weights)) => {
                operation.execute(input, selected, routing, weights, intermediate, output)
            },
            (Self::Ggml(experts), ExpertWeights::Ggml(weights)) => {
                experts.execute(input, selected, routing, weights, output)
            },
            _ => Err(Error::InvalidExecutionPlan(
                "routed-expert plan differs from layer weight storage",
            )),
        }
    }

    pub(in crate::backend) fn nvfp4_prequant_scale(&self) -> Option<DeviceBuffer<f32>> {
        match self {
            Self::NvFp4(experts) => experts.prequant_scale(),
            Self::Dense { .. } | Self::Ggml(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_nvfp4_prequantized_residual_shared(
        &mut self,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &ExpertWeights,
        residual: &DeviceBuffer<bf16>,
        shared: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weights) {
            (Self::NvFp4(experts), ExpertWeights::NvFp4 { .. }) => experts
                .execute_prequantized_residual_shared(
                    packed, scales, selected, routing, residual, shared, output,
                ),
            _ => Err(Error::InvalidExecutionPlan(
                "fused routed-expert output differs from layer storage",
            )),
        }
    }
}
