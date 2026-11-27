use mircuda::Stream;

use super::{
    super::{ClampedRoutedConfig, ClampedRoutedQkvLowering, scratch::ClampedRoutedScratch},
    linear::{ClampedRoutedLinear, ClampedRoutedLinearWeight, affine, phase},
};
use crate::{
    AffineQuantizedWeight, Bf16LinearPack, Bf16LinearPackWeights, CudaBackend, CudaTensor,
    DenseRole, GgmlBf16Linear, GgmlCheckpointWeight, Result, backend::linear::AffineProjection,
    kernels::ClampedRoutedKernels,
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
    Ggml(Box<[GgmlCheckpointWeight; 3]>),
    Mixed(Box<[ClampedRoutedLinearWeight; 3]>),
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
    Ggml {
        operations: Box<[GgmlBf16Linear; 3]>,
        weights: Box<[GgmlCheckpointWeight; 3]>,
        biases: [CudaTensor; 3],
    },
    Mixed {
        operations: Box<[ClampedRoutedLinear; 3]>,
        biases: [CudaTensor; 3],
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
                ClampedRoutedQkvLowering::PackedFused | ClampedRoutedQkvLowering::SeparateComposed,
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
            (
                ClampedRoutedQkvLowering::SeparateComposed,
                ClampedRoutedQkvProjections::Ggml(weights),
            ) => {
                let role = DenseRole::AttentionQkv;
                Ok(Self::Ggml {
                    operations: Box::new([
                        weights[0].prepare(backend, tokens, role)?,
                        weights[1].prepare(backend, tokens, role)?,
                        weights[2].prepare(backend, tokens, role)?,
                    ]),
                    weights: weights.clone(),
                    biases: weight.biases.clone(),
                })
            },
            (
                ClampedRoutedQkvLowering::SeparateComposed,
                ClampedRoutedQkvProjections::Mixed(weights),
            ) => {
                let role = DenseRole::AttentionQkv;
                Ok(Self::Mixed {
                    operations: Box::new([
                        ClampedRoutedLinear::new(
                            backend, tokens, config.hidden, outputs[0], role, &weights[0],
                        )?,
                        ClampedRoutedLinear::new(
                            backend, tokens, config.hidden, outputs[1], role, &weights[1],
                        )?,
                        ClampedRoutedLinear::new(
                            backend, tokens, config.hidden, outputs[2], role, &weights[2],
                        )?,
                    ]),
                    biases: weight.biases.clone(),
                })
            },
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
            Self::Ggml { operations, weights, biases } => {
                // One input: Q quantizes it, K and V reuse that.
                let input = &scratch.normalized;
                operations[0].execute_shared(input, &weights[0], &mut scratch.raw_query, false)?;
                operations[1].execute_shared(input, &weights[1], &mut scratch.raw_key, true)?;
                operations[2].execute_shared(input, &weights[2], &mut scratch.raw_value, true)?;
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
            Self::Mixed { operations, biases } => {
                operations[0].execute(&scratch.normalized, &mut scratch.raw_query)?;
                operations[1].execute(&scratch.normalized, &mut scratch.raw_key)?;
                operations[2].execute(&scratch.normalized, &mut scratch.raw_value)?;
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
