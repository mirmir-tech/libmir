use mircuda::{DeviceBuffer, bf16};

use super::{AffineGatedDeltaLayerConfig, checked};
use crate::{CudaBackend, Result};

/// Shape of one Gated Delta scratch set; layers with equal shapes share it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct GatedDeltaScratchKey {
    tokens: usize,
    mixed: usize,
    key: usize,
    value: usize,
    heads: usize,
    packed_qkv_gate: bool,
    packed_alpha_beta: bool,
}

impl GatedDeltaScratchKey {
    pub(super) fn new(
        config: AffineGatedDeltaLayerConfig,
        tokens: usize,
        packed_qkv_gate: bool,
        packed_alpha_beta: bool,
    ) -> Result<Self> {
        Ok(Self {
            tokens,
            mixed: config.mixed_width()?,
            key: config.key_width()?,
            value: config.value_width()?,
            heads: config.value_heads,
            packed_qkv_gate,
            packed_alpha_beta,
        })
    }
}

#[derive(Debug)]
pub(in crate::backend) struct GatedDeltaScratch {
    pub(super) packed_qkv_gate: Option<DeviceBuffer<bf16>>,
    pub(super) mixed: DeviceBuffer<bf16>,
    pub(super) convolved: DeviceBuffer<bf16>,
    pub(super) value: DeviceBuffer<bf16>,
    pub(super) normalized_query: DeviceBuffer<bf16>,
    pub(super) normalized_key: DeviceBuffer<bf16>,
    pub(super) gate: DeviceBuffer<bf16>,
    pub(super) alpha: DeviceBuffer<bf16>,
    pub(super) beta: DeviceBuffer<bf16>,
    pub(super) packed_alpha_beta: Option<DeviceBuffer<bf16>>,
    pub(super) recurrent: DeviceBuffer<bf16>,
    pub(super) gated: DeviceBuffer<bf16>,
}

impl GatedDeltaScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
        tokens: usize,
        packed_qkv_gate: bool,
        packed_alpha_beta: bool,
    ) -> Result<Self> {
        let mixed = checked(tokens, config.mixed_width()?)?;
        let key = checked(tokens, config.key_width()?)?;
        let value = checked(tokens, config.value_width()?)?;
        let heads = checked(tokens, config.value_heads)?;
        let paired_heads = checked(heads, 2)?;
        let packed = mixed
            .checked_add(value)
            .ok_or(crate::Error::InvalidDecoderKernel("packed Gated Delta size overflow"))?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            packed_qkv_gate: packed_qkv_gate.then(|| allocate(packed)).transpose()?,
            mixed: allocate(mixed)?,
            convolved: backend.inner.pool.allocate_zeroed(&backend.inner.stream, mixed)?,
            value: allocate(value)?,
            normalized_query: allocate(key)?,
            normalized_key: allocate(key)?,
            gate: allocate(value)?,
            alpha: allocate(heads)?,
            beta: allocate(heads)?,
            packed_alpha_beta: packed_alpha_beta.then(|| allocate(paired_heads)).transpose()?,
            recurrent: backend.inner.pool.allocate_zeroed(&backend.inner.stream, value)?,
            gated: allocate(value)?,
        })
    }
}
