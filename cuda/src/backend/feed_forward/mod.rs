mod dense;

pub(super) use dense::{DenseFeedForward, scratch::DenseScratchPool};
use mircuda::{DeviceBuffer, bf16};

use crate::{
    CudaAffineSharedExpertMoe, CudaAffineSharedExpertMoeExecution, ExecutionPhase, Result,
    kernels::{ElementwiseBf16, ShiftedRmsNorm},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerNormConfig {
    pub hidden_size: usize,
    pub rms_norm_epsilon: f32,
    pub norm_weight_shift: f32,
}

#[derive(Clone, Debug)]
pub(super) enum FeedForward {
    Dense(Box<DenseFeedForward>),
    SharedRouted(Box<CudaAffineSharedExpertMoe>),
}

#[derive(Debug)]
pub(super) enum FeedForwardExecution {
    Dense(Box<dense::DenseExecution>),
    SharedRouted(Box<CudaAffineSharedExpertMoeExecution>),
}

impl FeedForward {
    pub(super) fn prepare_phase(
        &self,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<FeedForwardExecution> {
        match self {
            Self::Dense(value) => {
                value.prepare(tokens).map(Box::new).map(FeedForwardExecution::Dense)
            },
            Self::SharedRouted(value) => value
                .prepare_phase(tokens, phase)
                .map(Box::new)
                .map(FeedForwardExecution::SharedRouted),
        }
    }
}

impl FeedForwardExecution {
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Dense(value) => value.execute(input, output),
            Self::SharedRouted(value) => value.execute(input, output),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_residual_norm(
        &mut self,
        norm: &ShiftedRmsNorm,
        input: &DeviceBuffer<bf16>,
        update: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        residual: &mut DeviceBuffer<bf16>,
        normalized: &mut DeviceBuffer<bf16>,
        ff_output: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        residual_add: &ElementwiseBf16,
    ) -> Result<()> {
        match self {
            Self::SharedRouted(value) => value.execute_residual_norm(
                norm, input, update, weight, residual, normalized, ff_output, output, residual_add,
            ),
            Self::Dense(value) => {
                let stream = value.stream();
                norm.execute_residual(&stream, input, update, weight, residual, normalized)?;
                value.execute(normalized, ff_output)?;
                residual_add.add(&stream, residual, ff_output, output)
            },
        }
    }
}
