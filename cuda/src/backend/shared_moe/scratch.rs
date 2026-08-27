use mircuda::{DeviceBuffer, bf16};

use super::{AffineSharedExpertMoeConfig, checked};
use crate::{CudaBackend, Result, kernels::scale_elements};

#[derive(Debug)]
pub(super) struct NvFp4InputScratch {
    pub(super) packed: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
}

#[derive(Debug)]
pub(super) struct AffineSharedMoeScratch {
    pub(super) routed_intermediate: DeviceBuffer<bf16>,
    pub(super) routed_output: DeviceBuffer<bf16>,
    pub(super) shared_gate: DeviceBuffer<bf16>,
    pub(super) shared_up: DeviceBuffer<bf16>,
    pub(super) shared_gate_up: DeviceBuffer<bf16>,
    pub(super) shared_intermediate: DeviceBuffer<bf16>,
    pub(super) shared_output: DeviceBuffer<bf16>,
    pub(super) shared_output_gate: DeviceBuffer<bf16>,
    pub(super) gated_shared_output: DeviceBuffer<bf16>,
    pub(super) nvfp4_input: Option<NvFp4InputScratch>,
}

impl AffineSharedMoeScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        tokens: usize,
    ) -> Result<Self> {
        let routed = checked(checked(tokens, config.top_k)?, config.routed_intermediate_size)?;
        let hidden = checked(tokens, config.hidden_size)?;
        let shared = checked(tokens, config.shared_intermediate_size)?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        let nvfp4_input = if config.hidden_size.is_multiple_of(64) {
            Some(NvFp4InputScratch {
                packed: backend.inner.pool.allocate(&backend.inner.stream, hidden / 2)?,
                scales: backend
                    .inner
                    .pool
                    .allocate(&backend.inner.stream, scale_elements(tokens, config.hidden_size)?)?,
            })
        } else {
            None
        };
        Ok(Self {
            routed_intermediate: allocate(routed)?,
            routed_output: allocate(hidden)?,
            shared_gate: allocate(shared)?,
            shared_up: allocate(shared)?,
            shared_gate_up: allocate(checked(shared, 2)?)?,
            shared_intermediate: allocate(shared)?,
            shared_output: allocate(hidden)?,
            shared_output_gate: allocate(tokens)?,
            gated_shared_output: allocate(hidden)?,
            nvfp4_input,
        })
    }
}
