use mircuda::{DeviceBuffer, bf16};

use super::super::weights::ClampedRoutedExpertWeights;
use crate::{
    CudaTensor, Error, Result, backend::tuning::ClampedMoeExecution, kernels::ClampedRoutedKernels,
};

pub(super) struct Candidate {
    pub(super) execution: ClampedMoeExecution,
    kernels: ClampedRoutedKernels,
}

impl Candidate {
    pub(super) const fn new(kernels: ClampedRoutedKernels, execution: ClampedMoeExecution) -> Self {
        Self { execution, kernels }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &self,
        stream: &mircuda::Stream,
        weights: &ClampedRoutedExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        activated: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match weights {
            ClampedRoutedExpertWeights::Native(weights) => {
                self.kernels.gate_up_native(
                    stream,
                    input,
                    u8s(&weights.gate_up_blocks)?,
                    u8s(&weights.gate_up_scales)?,
                    bf16s(&weights.gate_up_bias)?,
                    selected,
                    activated,
                )?;
                match self.execution {
                    ClampedMoeExecution::RouteParallel => self.kernels.down_routes_native(
                        stream,
                        activated,
                        u8s(&weights.down_blocks)?,
                        u8s(&weights.down_scales)?,
                        bf16s(&weights.down_bias)?,
                        selected,
                        routing,
                        partial,
                        output,
                    ),
                    ClampedMoeExecution::FusedReduce => self.kernels.down_native(
                        stream,
                        activated,
                        u8s(&weights.down_blocks)?,
                        u8s(&weights.down_scales)?,
                        bf16s(&weights.down_bias)?,
                        selected,
                        routing,
                        output,
                    ),
                }
            },
            ClampedRoutedExpertWeights::Mlx(weights) => {
                self.kernels.gate_up_mlx(
                    stream,
                    input,
                    u32s(&weights.gate_blocks)?,
                    u8s(&weights.gate_scales)?,
                    bf16s(&weights.gate_bias)?,
                    u32s(&weights.up_blocks)?,
                    u8s(&weights.up_scales)?,
                    bf16s(&weights.up_bias)?,
                    selected,
                    activated,
                )?;
                match self.execution {
                    ClampedMoeExecution::RouteParallel => self.kernels.down_routes_mlx(
                        stream,
                        activated,
                        u32s(&weights.down_blocks)?,
                        u8s(&weights.down_scales)?,
                        bf16s(&weights.down_bias)?,
                        selected,
                        routing,
                        partial,
                        output,
                    ),
                    ClampedMoeExecution::FusedReduce => self.kernels.down_mlx(
                        stream,
                        activated,
                        u32s(&weights.down_blocks)?,
                        u8s(&weights.down_scales)?,
                        bf16s(&weights.down_bias)?,
                        selected,
                        routing,
                        output,
                    ),
                }
            },
            ClampedRoutedExpertWeights::Dense(_) => {
                Err(Error::InvalidExecutionPlan("dense experts cannot use clamped MXFP4 execution"))
            },
        }
    }
}

fn bf16s(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}

fn u8s(tensor: &CudaTensor) -> Result<&DeviceBuffer<u8>> {
    tensor.as_u8().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "U8",
    })
}

fn u32s(tensor: &CudaTensor) -> Result<&DeviceBuffer<u32>> {
    tensor.as_u32().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "U32",
    })
}
