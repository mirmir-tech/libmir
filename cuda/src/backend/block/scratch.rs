use mircuda::{DeviceBuffer, bf16};

use super::CudaBackend;
use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct BlockScratch {
    pub(super) attention: DeviceBuffer<bf16>,
    pub(super) attention_norm: DeviceBuffer<bf16>,
    pub(super) hidden: DeviceBuffer<bf16>,
    pub(super) normalized: DeviceBuffer<bf16>,
    pub(super) dense_gate_up: DeviceBuffer<bf16>,
    pub(super) dense_activated: DeviceBuffer<bf16>,
    pub(super) dense: DeviceBuffer<bf16>,
    pub(super) expert: DeviceBuffer<bf16>,
    pub(super) expert_norm: DeviceBuffer<bf16>,
    pub(super) feed_forward: DeviceBuffer<bf16>,
    pub(super) feed_forward_norm: DeviceBuffer<bf16>,
    pub(super) residual: DeviceBuffer<bf16>,
}

impl BlockScratch {
    pub(super) fn new(backend: &CudaBackend, hidden: usize, dense: usize) -> Result<Self> {
        let allocate =
            |elements| backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
        let dense_pair = dense
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("dense pair scratch size overflow"))?;
        Ok(Self {
            attention: allocate(hidden)?,
            attention_norm: allocate(hidden)?,
            hidden: allocate(hidden)?,
            normalized: allocate(hidden)?,
            dense_gate_up: allocate(dense_pair)?,
            dense_activated: allocate(dense)?,
            dense: allocate(hidden)?,
            expert: allocate(hidden)?,
            expert_norm: allocate(hidden)?,
            feed_forward: allocate(hidden)?,
            feed_forward_norm: allocate(hidden)?,
            residual: allocate(hidden)?,
        })
    }
}
