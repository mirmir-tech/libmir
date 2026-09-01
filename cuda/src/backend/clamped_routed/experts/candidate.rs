use mircuda::{DeviceBuffer, bf16};

use super::{
    super::weights::{ClampedRoutedExpertWeights, NativeExpertWeights},
    ClampedRoutedConfig,
    marlin::MarlinMxFp4Candidate,
};
use crate::{
    CudaBackend, CudaTensor, Error, Result, backend::tuning::ClampedMoeExecution,
    kernels::ClampedRoutedKernels,
};

pub(super) struct Candidate {
    pub(super) execution: ClampedMoeExecution,
    plan: Plan,
}

enum Plan {
    Portable(ClampedRoutedKernels),
    Marlin(MarlinMxFp4Candidate),
}

impl Candidate {
    pub(super) const fn portable(
        kernels: ClampedRoutedKernels,
        execution: ClampedMoeExecution,
    ) -> Self {
        Self { execution, plan: Plan::Portable(kernels) }
    }

    pub(super) fn marlin(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        tokens: usize,
        weights: &NativeExpertWeights,
        execution: ClampedMoeExecution,
    ) -> Result<Self> {
        Ok(Self {
            execution,
            plan: Plan::Marlin(MarlinMxFp4Candidate::new(
                backend, config, tokens, weights, execution,
            )?),
        })
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
        match (&self.plan, weights) {
            (Plan::Marlin(plan), ClampedRoutedExpertWeights::Native(weights)) => {
                plan.execute(weights, input, selected, routing, output)
            },
            (Plan::Marlin(_), _) => Err(Error::InvalidExecutionPlan(
                "MXFP4 Marlin requires native clamped expert weights",
            )),
            (Plan::Portable(kernels), ClampedRoutedExpertWeights::Native(weights)) => {
                execute_native(
                    kernels, self.execution, stream, weights, input, selected, routing, activated,
                    partial, output,
                )
            },
            (Plan::Portable(kernels), ClampedRoutedExpertWeights::Mlx(weights)) => {
                kernels.gate_up_mlx(
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
                    ClampedMoeExecution::RouteParallel => kernels.down_routes_mlx(
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
                    ClampedMoeExecution::FusedReduce => kernels.down_mlx(
                        stream,
                        activated,
                        u32s(&weights.down_blocks)?,
                        u8s(&weights.down_scales)?,
                        bf16s(&weights.down_bias)?,
                        selected,
                        routing,
                        output,
                    ),
                    _ => invalid_portable(),
                }
            },
            (Plan::Portable(_), ClampedRoutedExpertWeights::Dense(_)) => {
                Err(Error::InvalidExecutionPlan("dense experts cannot use clamped MXFP4 execution"))
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_native(
    kernels: &ClampedRoutedKernels,
    execution: ClampedMoeExecution,
    stream: &mircuda::Stream,
    weights: &NativeExpertWeights,
    input: &DeviceBuffer<bf16>,
    selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>,
    activated: &mut DeviceBuffer<bf16>,
    partial: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    kernels.gate_up_native(
        stream,
        input,
        u8s(&weights.gate_up_blocks)?,
        u8s(&weights.gate_up_scales)?,
        bf16s(&weights.gate_up_bias)?,
        selected,
        activated,
    )?;
    match execution {
        ClampedMoeExecution::RouteParallel => kernels.down_routes_native(
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
        ClampedMoeExecution::FusedReduce => kernels.down_native(
            stream,
            activated,
            u8s(&weights.down_blocks)?,
            u8s(&weights.down_scales)?,
            bf16s(&weights.down_bias)?,
            selected,
            routing,
            output,
        ),
        _ => invalid_portable(),
    }
}

fn invalid_portable() -> Result<()> {
    Err(Error::InvalidExecutionPlan(
        "Marlin execution cannot use a portable clamped plan",
    ))
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
