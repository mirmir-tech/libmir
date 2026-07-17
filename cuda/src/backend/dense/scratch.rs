use mircuda::{DeviceBuffer, bf16};

use super::{CudaBackend, DenseSwiGluConfig};
use crate::{Error, Result};

pub(super) struct DenseScratch {
    pub attention: DeviceBuffer<bf16>,
    pub residual: DeviceBuffer<bf16>,
    pub normalized: DeviceBuffer<bf16>,
    pub gate_up: DeviceBuffer<bf16>,
    pub gate_up_separate: [DeviceBuffer<bf16>; 2],
    pub activated: DeviceBuffer<bf16>,
    pub mlp: DeviceBuffer<bf16>,
}

impl DenseScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        tokens: usize,
    ) -> Result<Self> {
        let hidden = elements(tokens, config.attention.hidden_size)?;
        let intermediate = elements(tokens, config.intermediate_size)?;
        let gate_up = intermediate
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("dense gate/up scratch overflow"))?;
        let allocate = |size| backend.inner.pool.allocate(&backend.inner.stream, size);
        Ok(Self {
            attention: allocate(hidden)?,
            residual: allocate(hidden)?,
            normalized: allocate(hidden)?,
            gate_up: allocate(gate_up)?,
            gate_up_separate: [allocate(intermediate)?, allocate(intermediate)?],
            activated: allocate(intermediate)?,
            mlp: allocate(hidden)?,
        })
    }
}

pub(super) fn elements(tokens: usize, width: usize) -> Result<usize> {
    tokens
        .checked_mul(width)
        .ok_or(Error::InvalidDecoderKernel("dense SwiGLU shape overflow"))
}
