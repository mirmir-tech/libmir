use mircuda::{DeviceBuffer, Stream, bf16};

use super::super::{ClampedRoutedConfig, ClampedRoutedQkvLowering, scratch::ClampedRoutedScratch};
use crate::{
    AffineQuantizedWeight, Bf16LinearPack, Bf16LinearPackWeights, Bf16Projection, CudaBackend,
    CudaTensor, DensePlanRequest, DenseRole, ExecutionPhase, Result,
    backend::linear::AffineProjection, kernels::ClampedRoutedKernels,
};

#[derive(Clone)]
pub(in crate::backend::clamped_routed) struct ClampedRoutedQkvWeight {
    pub projections: ClampedRoutedQkvProjections,
    pub biases: [CudaTensor; 3],
}

#[derive(Clone)]
pub(in crate::backend::clamped_routed) enum ClampedRoutedQkvProjections {
    Native(Bf16LinearPackWeights<3>),
    Mlx(Box<[AffineQuantizedWeight; 3]>),
}

pub(in crate::backend::clamped_routed) enum ClampedRoutedQkv {
    Native {
        operation: Bf16LinearPack<3>,
        weights: Bf16LinearPackWeights<3>,
        biases: [CudaTensor; 3],
    },
    Mlx {
        operations: Box<[AffineProjection; 3]>,
        weights: Box<[AffineQuantizedWeight; 3]>,
        biases: [CudaTensor; 3],
    },
}

#[derive(Clone)]
pub(in crate::backend::clamped_routed) enum ClampedRoutedLinearWeight {
    Native(CudaTensor),
    Mlx(AffineQuantizedWeight),
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
}

impl ClampedRoutedQkv {
    pub(in crate::backend::clamped_routed) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        tokens: usize,
        lowering: ClampedRoutedQkvLowering,
        weight: &ClampedRoutedQkvWeight,
    ) -> Result<Self> {
        let outputs = [
            config.query_heads * config.head_dim,
            config.kv_heads * config.head_dim,
            config.kv_heads * config.head_dim,
        ];
        match (lowering, &weight.projections) {
            (
                ClampedRoutedQkvLowering::PackedFused,
                ClampedRoutedQkvProjections::Native(weights),
            ) => Ok(Self::Native {
                operation: Bf16LinearPack::new(
                    backend,
                    phase(tokens),
                    tokens,
                    config.hidden,
                    outputs,
                )?,
                weights: weights.clone(),
                biases: weight.biases.clone(),
            }),
            (
                ClampedRoutedQkvLowering::SeparateComposed,
                ClampedRoutedQkvProjections::Mlx(weights),
            ) => Ok(Self::Mlx {
                operations: Box::new([
                    affine(backend, tokens, config.hidden, outputs[0], &weights[0])?,
                    affine(backend, tokens, config.hidden, outputs[1], &weights[1])?,
                    affine(backend, tokens, config.hidden, outputs[2], &weights[2])?,
                ]),
                weights: weights.clone(),
                biases: weight.biases.clone(),
            }),
            _ => Err(crate::Error::InvalidExecutionPlan(
                "clamped-routed QKV lowering differs from bound storage",
            )),
        }
    }

    pub(in crate::backend::clamped_routed) fn execute(
        &mut self,
        kernels: &ClampedRoutedKernels,
        stream: &Stream,
        scratch: &mut ClampedRoutedScratch,
    ) -> Result<()> {
        match self {
            Self::Native { operation, weights, biases } => {
                operation.execute(&scratch.normalized, weights, &mut scratch.packed_qkv)?;
                kernels.qkv_native(
                    stream,
                    &scratch.packed_qkv,
                    biases,
                    &mut scratch.query,
                    &mut scratch.key,
                    &mut scratch.value,
                    &scratch.rope_sines,
                    &scratch.rope_cosines,
                    scratch.rope_concentration,
                )
            },
            Self::Mlx { operations, weights, biases } => {
                operations[0].execute(&scratch.normalized, &weights[0], &mut scratch.raw_query)?;
                operations[1].execute(&scratch.normalized, &weights[1], &mut scratch.raw_key)?;
                operations[2].execute(&scratch.normalized, &weights[2], &mut scratch.raw_value)?;
                kernels.qkv_mlx(
                    stream,
                    [&scratch.raw_query, &scratch.raw_key, &scratch.raw_value],
                    biases,
                    &mut scratch.query,
                    &mut scratch.key,
                    &mut scratch.value,
                    &scratch.rope_sines,
                    &scratch.rope_cosines,
                    scratch.rope_concentration,
                )
            },
        }
    }
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
        }
    }
}

fn affine(
    backend: &CudaBackend,
    tokens: usize,
    input: usize,
    output: usize,
    weight: &AffineQuantizedWeight,
) -> Result<AffineProjection> {
    let config = weight.infer_config(1, input, output)?;
    AffineProjection::new(backend, tokens, input, output, config.group_size, config.bits, weight)
}

const fn phase(tokens: usize) -> ExecutionPhase {
    if tokens == 1 {
        ExecutionPhase::Decode
    } else {
        ExecutionPhase::Prefill
    }
}
