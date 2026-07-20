use mircuda::{DeviceBuffer, bf16};
use models::layout::PooledVisionConfig;

use super::super::super::CudaBackend;
use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct PooledScratch {
    pub hidden_a: DeviceBuffer<bf16>,
    pub hidden_b: DeviceBuffer<bf16>,
    pub normalized: DeviceBuffer<bf16>,
    pub query: DeviceBuffer<bf16>,
    pub query_rope: DeviceBuffer<bf16>,
    pub key: DeviceBuffer<bf16>,
    pub key_rope: DeviceBuffer<bf16>,
    pub value: DeviceBuffer<bf16>,
    pub intermediate_a: DeviceBuffer<bf16>,
    pub intermediate_b: DeviceBuffer<bf16>,
    pub intermediate_c: DeviceBuffer<bf16>,
    pub pooled_a: DeviceBuffer<bf16>,
    pub pooled_b: DeviceBuffer<bf16>,
    pub output: DeviceBuffer<bf16>,
}

impl PooledScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PooledVisionConfig,
        tokens: usize,
        pooled_tokens: usize,
    ) -> Result<Self> {
        let hidden = elements(tokens, config.hidden_size)?;
        let key_value = elements(tokens, config.num_key_value_heads * config.head_dim)?;
        let intermediate = elements(tokens, config.intermediate_size)?;
        let pooled = elements(pooled_tokens, config.hidden_size)?;
        let output = elements(pooled_tokens, config.output_hidden_size)?;
        let allocate = |size| backend.inner.pool.allocate(&backend.inner.stream, size);
        Ok(Self {
            hidden_a: allocate(hidden)?,
            hidden_b: allocate(hidden)?,
            normalized: allocate(hidden)?,
            query: allocate(hidden)?,
            query_rope: allocate(hidden)?,
            key: allocate(key_value)?,
            key_rope: allocate(key_value)?,
            value: allocate(key_value)?,
            intermediate_a: allocate(intermediate)?,
            intermediate_b: allocate(intermediate)?,
            intermediate_c: allocate(intermediate)?,
            pooled_a: allocate(pooled)?,
            pooled_b: allocate(pooled)?,
            output: allocate(output)?,
        })
    }
}

fn elements(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidVisionKernel("vision scratch overflow"))
}
