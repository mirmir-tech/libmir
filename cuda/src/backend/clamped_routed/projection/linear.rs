use mircuda::{DeviceBuffer, bf16};

use crate::{
    AffineQuantizedWeight, Bf16Projection, CudaBackend, CudaTensor, DensePlanRequest, DenseRole,
    ExecutionPhase, GgmlBf16Linear, GgmlCheckpointWeight, Result,
    backend::linear::AffineProjection,
};

#[derive(Clone)]
pub(in crate::backend::clamped_routed) enum ClampedRoutedLinearWeight {
    Native(CudaTensor),
    Mlx(AffineQuantizedWeight),
    Ggml(GgmlCheckpointWeight),
}

pub(in crate::backend::clamped_routed) enum ClampedRoutedLinear {
    Native {
        operation: Bf16Projection,
        weight: CudaTensor,
    },
    Mlx {
        operation: AffineProjection,
        weight: AffineQuantizedWeight,
    },
    Ggml {
        operation: Box<GgmlBf16Linear>,
        weight: GgmlCheckpointWeight,
    },
}

impl ClampedRoutedLinear {
    pub(in crate::backend::clamped_routed) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        role: DenseRole,
        weight: &ClampedRoutedLinearWeight,
    ) -> Result<Self> {
        match weight {
            ClampedRoutedLinearWeight::Native(weight) => Ok(Self::Native {
                operation: backend.prepare_bf16_projection(DensePlanRequest {
                    phase: phase(tokens),
                    role,
                    tokens,
                    input_features: input,
                    output_features: output,
                })?,
                weight: weight.clone(),
            }),
            ClampedRoutedLinearWeight::Mlx(weight) => Ok(Self::Mlx {
                operation: affine(backend, tokens, input, output, weight)?,
                weight: weight.clone(),
            }),
            ClampedRoutedLinearWeight::Ggml(weight) => Ok(Self::Ggml {
                operation: Box::new(weight.prepare(backend, tokens, role)?),
                weight: weight.clone(),
            }),
        }
    }

    pub(in crate::backend::clamped_routed) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Native { operation, weight } => operation.execute(input, weight, output),
            Self::Mlx { operation, weight } => operation.execute(input, weight, output),
            Self::Ggml { operation, weight } => operation.execute(input, weight, output),
        }
    }
}

pub(super) fn affine(
    backend: &CudaBackend,
    tokens: usize,
    input: usize,
    output: usize,
    weight: &AffineQuantizedWeight,
) -> Result<AffineProjection> {
    let config = weight.infer_config(1, input, output)?;
    AffineProjection::new(backend, tokens, input, output, config.group_size, config.bits, weight)
}

pub(super) const fn phase(tokens: usize) -> ExecutionPhase {
    if tokens == 1 {
        ExecutionPhase::Decode
    } else {
        ExecutionPhase::Prefill
    }
}
