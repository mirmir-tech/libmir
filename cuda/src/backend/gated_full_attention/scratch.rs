use mircuda::{DeviceBuffer, bf16};

use super::{AffineGatedFullAttentionConfig, checked};
use crate::{CudaBackend, Result};

#[derive(Debug)]
pub(super) struct GatedAttentionScratch {
    pub(super) packed_qkv: Option<DeviceBuffer<bf16>>,
    pub(super) query_projected: DeviceBuffer<bf16>,
    pub(super) query: DeviceBuffer<bf16>,
    pub(super) gate: DeviceBuffer<bf16>,
    pub(super) normalized_query: DeviceBuffer<bf16>,
    pub(super) rotated_query: DeviceBuffer<bf16>,
    pub(super) key: DeviceBuffer<bf16>,
    pub(super) normalized_key: DeviceBuffer<bf16>,
    pub(super) rotated_key: DeviceBuffer<bf16>,
    pub(super) value: DeviceBuffer<bf16>,
    pub(super) attended: DeviceBuffer<bf16>,
    pub(super) gated: DeviceBuffer<bf16>,
}

impl GatedAttentionScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedFullAttentionConfig,
        tokens: usize,
        packed_qkv: bool,
    ) -> Result<Self> {
        let query = checked(tokens, config.query_width()?)?;
        let key_value = checked(tokens, config.key_value_width()?)?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        let packed = checked(query, 2)?
            .checked_add(checked(key_value, 2)?)
            .ok_or(crate::Error::InvalidDecoderKernel("packed attention size overflow"))?;
        Ok(Self {
            packed_qkv: packed_qkv.then(|| allocate(packed)).transpose()?,
            query_projected: allocate(checked(query, 2)?)?,
            query: allocate(query)?,
            gate: allocate(query)?,
            normalized_query: allocate(query)?,
            rotated_query: allocate(query)?,
            key: allocate(key_value)?,
            normalized_key: allocate(key_value)?,
            rotated_key: allocate(key_value)?,
            value: allocate(key_value)?,
            attended: allocate(query)?,
            gated: allocate(query)?,
        })
    }
}
