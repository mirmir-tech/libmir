use mircuda::{DeviceBuffer, bf16};
use models::layout::SpatialMergeVisionConfig;

use super::super::super::CudaBackend;
use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct SpatialMergeScratch {
    pub patches: DeviceBuffer<bf16>,
    pub hidden_a: DeviceBuffer<bf16>,
    pub hidden_b: DeviceBuffer<bf16>,
    pub normalized: DeviceBuffer<bf16>,
    pub qkv: DeviceBuffer<bf16>,
    pub query: DeviceBuffer<bf16>,
    pub query_rope: DeviceBuffer<bf16>,
    pub key: DeviceBuffer<bf16>,
    pub key_rope: DeviceBuffer<bf16>,
    pub value: DeviceBuffer<bf16>,
    pub intermediate_a: DeviceBuffer<bf16>,
    pub intermediate_b: DeviceBuffer<bf16>,
    pub output: DeviceBuffer<bf16>,
}

impl SpatialMergeScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &SpatialMergeVisionConfig,
        tokens: usize,
        soft_tokens: usize,
    ) -> Result<Self> {
        let hidden = elements(tokens, config.hidden_size)?;
        let intermediate = elements(tokens, config.intermediate_size)?;
        let output = elements(soft_tokens, config.output_hidden_size)?;
        let patch_width = elements(
            elements(config.in_channels, config.temporal_patch_size)?,
            elements(config.patch_size, config.patch_size)?,
        )?;
        let allocate = |size| backend.inner.pool.allocate(&backend.inner.stream, size);
        Ok(Self {
            patches: allocate(elements(tokens, patch_width)?)?,
            hidden_a: allocate(hidden)?,
            hidden_b: allocate(hidden)?,
            normalized: allocate(hidden)?,
            qkv: allocate(elements(hidden, 3)?)?,
            query: allocate(hidden)?,
            query_rope: allocate(hidden)?,
            key: allocate(hidden)?,
            key_rope: allocate(hidden)?,
            value: allocate(hidden)?,
            intermediate_a: allocate(intermediate)?,
            intermediate_b: allocate(intermediate)?,
            output: allocate(output)?,
        })
    }
}

fn elements(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidVisionKernel("spatial-merge scratch overflow"))
}
