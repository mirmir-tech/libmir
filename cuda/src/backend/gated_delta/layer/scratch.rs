use mircuda::{DeviceBuffer, bf16};

use super::{AffineGatedDeltaLayerConfig, checked};
use crate::{CudaBackend, Result};

#[derive(Debug)]
pub(super) struct GatedDeltaScratch {
    pub(super) packed_qkv_gate: Option<DeviceBuffer<bf16>>,
    pub(super) mixed: DeviceBuffer<bf16>,
    pub(super) convolved: DeviceBuffer<bf16>,
    pub(super) value: DeviceBuffer<bf16>,
    pub(super) normalized_query: DeviceBuffer<bf16>,
    pub(super) normalized_key: DeviceBuffer<bf16>,
    pub(super) gate: DeviceBuffer<bf16>,
    pub(super) alpha: DeviceBuffer<bf16>,
    pub(super) beta: DeviceBuffer<bf16>,
    pub(super) recurrent: DeviceBuffer<bf16>,
    pub(super) gated: DeviceBuffer<bf16>,
}

impl GatedDeltaScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
        tokens: usize,
        packed_qkv_gate: bool,
    ) -> Result<Self> {
        let mixed = checked(tokens, config.mixed_width()?)?;
        let key = checked(tokens, config.key_width()?)?;
        let value = checked(tokens, config.value_width()?)?;
        let heads = checked(tokens, config.value_heads)?;
        let packed = mixed
            .checked_add(value)
            .ok_or(crate::Error::InvalidDecoderKernel("packed Gated Delta size overflow"))?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            packed_qkv_gate: packed_qkv_gate.then(|| allocate(packed)).transpose()?,
            mixed: allocate(mixed)?,
            convolved: allocate(mixed)?,
            value: allocate(value)?,
            normalized_query: allocate(key)?,
            normalized_key: allocate(key)?,
            gate: allocate(value)?,
            alpha: allocate(heads)?,
            beta: allocate(heads)?,
            recurrent: allocate(value)?,
            gated: allocate(value)?,
        })
    }
}
