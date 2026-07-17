use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result};

#[derive(Debug)]
pub(super) struct PrefillBlockScratch {
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

impl PrefillBlockScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        hidden: usize,
        dense: usize,
    ) -> Result<Self> {
        let hidden = tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("prefill hidden scratch size overflow"))?;
        let dense = tokens
            .checked_mul(dense)
            .ok_or(Error::InvalidDecoderKernel("prefill dense scratch size overflow"))?;
        let dense_pair = dense
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("prefill dense pair size overflow"))?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
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
