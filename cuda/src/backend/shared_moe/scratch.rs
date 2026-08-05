use mircuda::{DeviceBuffer, bf16};

use super::{AffineSharedExpertMoeConfig, checked};
use crate::{CudaBackend, Result};

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
        })
    }
}
