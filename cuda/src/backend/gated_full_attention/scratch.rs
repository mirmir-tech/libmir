use mircuda::{DeviceBuffer, bf16};

use super::{AffineGatedFullAttentionConfig, checked};
use crate::{CudaBackend, Result};

/// Shape of one gated attention scratch set; layers with equal shapes share it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct GatedAttentionScratchKey {
    tokens: usize,
    query: usize,
    key_value: usize,
    packed_qkv: bool,
}

impl GatedAttentionScratchKey {
    pub(super) fn new(
        config: AffineGatedFullAttentionConfig,
        tokens: usize,
        packed_qkv: bool,
    ) -> Result<Self> {
        Ok(Self {
            tokens,
            query: config.query_width()?,
            key_value: config.key_value_width()?,
            packed_qkv,
        })
    }
}

#[derive(Debug)]
pub(in crate::backend) struct GatedAttentionScratch {
    pub(super) packed_qkv: Option<DeviceBuffer<bf16>>,
    pub(super) query_projected: DeviceBuffer<bf16>,
    pub(super) gate: DeviceBuffer<bf16>,
    pub(super) rotated_query: DeviceBuffer<bf16>,
    pub(super) key: DeviceBuffer<bf16>,
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
            gate: allocate(query)?,
            rotated_query: allocate(query)?,
            key: allocate(key_value)?,
            rotated_key: allocate(key_value)?,
            value: allocate(key_value)?,
            attended: backend.inner.pool.allocate_zeroed(&backend.inner.stream, query)?,
            gated: allocate(query)?,
        })
    }
}
