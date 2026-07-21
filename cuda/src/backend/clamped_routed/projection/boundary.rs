use mircuda::{DeviceBuffer, bf16};

use super::super::ClampedRoutedConfig;
use crate::{
    AffineQuantizedEmbedding, AffineQuantizedWeight, Bf16Embedding, Bf16Projection,
    CudaAffineOutputHead, CudaBackend, CudaTensor, DensePlanRequest, DenseRole, Error,
    ExecutionPhase, Result,
};

#[derive(Clone)]
pub(in crate::backend::clamped_routed) enum ClampedRoutedBoundaryProjection {
    Native(CudaTensor),
    Mlx(AffineQuantizedWeight),
}

pub(in crate::backend::clamped_routed) enum ClampedRoutedEmbedding {
    Native(Bf16Embedding),
    Mlx(AffineQuantizedEmbedding),
}

pub(in crate::backend::clamped_routed) enum ClampedRoutedOutput {
    Native(Bf16Projection),
    Mlx(Box<CudaAffineOutputHead>),
}

impl ClampedRoutedEmbedding {
    pub(in crate::backend::clamped_routed) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        weight: &ClampedRoutedBoundaryProjection,
    ) -> Result<Self> {
        match weight {
            ClampedRoutedBoundaryProjection::Native(_) => backend
                .prepare_bf16_embedding(config.vocab, config.hidden, 1.0)
                .map(Self::Native),
            ClampedRoutedBoundaryProjection::Mlx(weight) => {
                let quantized = weight.infer_config(1, config.hidden, config.vocab)?;
                backend.prepare_affine_embedding(quantized, 1.0).map(Self::Mlx)
            },
        }
    }

    pub(in crate::backend::clamped_routed) fn execute_batch(
        &self,
        tokens: &DeviceBuffer<u32>,
        count: usize,
        weight: &ClampedRoutedBoundaryProjection,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weight) {
            (Self::Native(operation), ClampedRoutedBoundaryProjection::Native(weight)) => {
                operation.execute_batch(tokens, 0, count, weight, output)
            },
            (Self::Mlx(operation), ClampedRoutedBoundaryProjection::Mlx(weight)) => {
                operation.execute_batch(tokens, 0, count, weight.tensors(), output)
            },
            _ => Err(Error::InvalidExecutionPlan("clamped-routed embedding layout changed")),
        }
    }

    pub(in crate::backend::clamped_routed) fn validate_token(&self, token: u32) -> Result<()> {
        match self {
            Self::Native(operation) => operation.validate_token(token),
            Self::Mlx(operation) => operation.validate_token(token),
        }
    }
}

impl ClampedRoutedOutput {
    pub(in crate::backend::clamped_routed) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        weight: &ClampedRoutedBoundaryProjection,
    ) -> Result<Self> {
        match weight {
            ClampedRoutedBoundaryProjection::Native(_) => backend
                .prepare_bf16_projection(DensePlanRequest {
                    phase: ExecutionPhase::Decode,
                    role: DenseRole::OutputHead,
                    tokens: 1,
                    input_features: config.hidden,
                    output_features: config.vocab,
                })
                .map(Self::Native),
            ClampedRoutedBoundaryProjection::Mlx(weight) => {
                CudaAffineOutputHead::from_weight(backend, config.hidden, config.vocab, weight)
                    .map(Box::new)
                    .map(Self::Mlx)
            },
        }
    }

    pub(in crate::backend::clamped_routed) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &ClampedRoutedBoundaryProjection,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weight) {
            (Self::Native(operation), ClampedRoutedBoundaryProjection::Native(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Mlx(operation), ClampedRoutedBoundaryProjection::Mlx(_)) => {
                operation.execute(input, output)
            },
            _ => Err(Error::InvalidExecutionPlan("clamped-routed output layout changed")),
        }
    }
}
