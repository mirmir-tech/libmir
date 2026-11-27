pub(in crate::backend) use layout::ProjectionLayout;

use super::{
    AffineProjection, AffineQuantizedWeight, Bf16Projection, DirectFp8Bf16Linear,
    DirectFp8CheckpointWeight, GgmlBf16Linear, GgmlCheckpointWeight, MxFp4Bf16Linear,
    MxFp4CheckpointWeight, MxFp8Bf16Linear, MxFp8CheckpointWeight, NvFp4Bf16Linear,
    NvFp4LinearWeight, NvFp4WeightOnlyBf16Linear, NvFp4WeightOnlyWeight, PackedIntegerBf16Linear,
    PackedIntegerWeight, validate_dense,
};
use crate::{CudaBackend, CudaTensor, DensePlanRequest, DenseRole, ExecutionPhase, Result};

mod execution;
mod layout;
mod load;
mod nvfp4;
mod pack;
mod validate;

#[derive(Clone, Debug)]
pub enum CheckpointProjectionWeight {
    Affine(AffineQuantizedWeight),
    Dense(CudaTensor),
    DirectFp8(DirectFp8CheckpointWeight),
    MxFp4(MxFp4CheckpointWeight),
    Ggml(GgmlCheckpointWeight),
    MxFp8(MxFp8CheckpointWeight),
    NvFp4(NvFp4LinearWeight),
    NvFp4WeightOnly(NvFp4WeightOnlyWeight),
    PackedInteger(PackedIntegerWeight),
}
#[derive(Debug)]
pub(in crate::backend) enum CheckpointProjection {
    Affine {
        operation: AffineProjection,
        weight: AffineQuantizedWeight,
    },
    Dense {
        operation: Bf16Projection,
        weight: CudaTensor,
    },
    DirectFp8 {
        operation: DirectFp8Bf16Linear,
        weight: DirectFp8CheckpointWeight,
    },
    MxFp4 {
        operation: MxFp4Bf16Linear,
        weight: MxFp4CheckpointWeight,
    },
    Ggml {
        operation: Box<GgmlBf16Linear>,
        weight: GgmlCheckpointWeight,
    },
    MxFp8 {
        operation: MxFp8Bf16Linear,
        weight: MxFp8CheckpointWeight,
    },
    NvFp4 {
        operation: NvFp4Bf16Linear,
    },
    NvFp4WeightOnly {
        operation: NvFp4WeightOnlyBf16Linear,
    },
    PackedInteger {
        operation: PackedIntegerBf16Linear,
        weight: PackedIntegerWeight,
    },
}
impl CheckpointProjection {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        role: DenseRole,
        weights: &CheckpointProjectionWeight,
    ) -> Result<Self> {
        match weights {
            CheckpointProjectionWeight::Affine(weights) => {
                let format = weights.infer_config(1, input, output)?;
                Ok(Self::Affine {
                    operation: AffineProjection::new(
                        backend,
                        tokens,
                        input,
                        output,
                        format.group_size,
                        format.bits,
                        weights,
                    )?,
                    weight: weights.clone(),
                })
            },
            CheckpointProjectionWeight::Dense(weight) => {
                validate_dense(weight, 1, input, output)?;
                Ok(Self::Dense {
                    operation: backend.prepare_bf16_projection(DensePlanRequest {
                        phase: if tokens == 1 {
                            ExecutionPhase::Decode
                        } else {
                            ExecutionPhase::Prefill
                        },
                        role,
                        tokens,
                        input_features: input,
                        output_features: output,
                    })?,
                    weight: weight.clone(),
                })
            },
            CheckpointProjectionWeight::DirectFp8(weight) => Ok(Self::DirectFp8 {
                operation: weight.prepare(backend, tokens)?,
                weight: weight.clone(),
            }),
            CheckpointProjectionWeight::MxFp4(weight) => Ok(Self::MxFp4 {
                operation: weight.prepare(backend, tokens)?,
                weight: weight.clone(),
            }),
            CheckpointProjectionWeight::Ggml(weight) => Ok(Self::Ggml {
                operation: Box::new(weight.prepare(backend, tokens, role)?),
                weight: weight.clone(),
            }),
            CheckpointProjectionWeight::MxFp8(weight) => Ok(Self::MxFp8 {
                operation: weight.prepare(backend, tokens)?,
                weight: weight.clone(),
            }),
            CheckpointProjectionWeight::NvFp4(weight) => Ok(Self::NvFp4 {
                operation: NvFp4Bf16Linear::from_weight(backend, tokens, weight.clone())?,
            }),
            CheckpointProjectionWeight::NvFp4WeightOnly(weight) => Ok(Self::NvFp4WeightOnly {
                operation: NvFp4WeightOnlyBf16Linear::new(backend, tokens, role, weight.clone())?,
            }),
            CheckpointProjectionWeight::PackedInteger(weight) => Ok(Self::PackedInteger {
                operation: PackedIntegerBf16Linear::new(backend, tokens, input, output, weight)?,
                weight: weight.clone(),
            }),
        }
    }
}
