use mircuda::{DeviceBuffer, bf16};

use super::ClampedRoutedConfig;
use crate::{CudaBackend, Error, Result};

pub(super) struct ClampedRoutedScratch {
    pub normalized: DeviceBuffer<bf16>,
    pub packed_qkv: DeviceBuffer<bf16>,
    pub raw_query: DeviceBuffer<bf16>,
    pub raw_key: DeviceBuffer<bf16>,
    pub raw_value: DeviceBuffer<bf16>,
    pub query: DeviceBuffer<bf16>,
    pub key: DeviceBuffer<bf16>,
    pub value: DeviceBuffer<bf16>,
    pub attended: DeviceBuffer<bf16>,
    pub projected: DeviceBuffer<bf16>,
    pub biased: DeviceBuffer<bf16>,
    pub residual: DeviceBuffer<bf16>,
    pub router: DeviceBuffer<bf16>,
    pub router_biased: DeviceBuffer<bf16>,
    pub selected: DeviceBuffer<u32>,
    pub routing: DeviceBuffer<bf16>,
    pub activated: DeviceBuffer<bf16>,
    pub moe: DeviceBuffer<bf16>,
}

impl ClampedRoutedScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        tokens: usize,
    ) -> Result<Self> {
        let bf16 = |elements| backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
        let hidden = product(tokens, config.hidden)?;
        let query = product(tokens, product(config.query_heads, config.head_dim)?)?;
        let kv = product(tokens, product(config.kv_heads, config.head_dim)?)?;
        let packed = query
            .checked_add(2 * kv)
            .ok_or(Error::InvalidDecoderKernel("clamped-routed QKV scratch overflow"))?;
        let routes = product(tokens, config.top_k)?;
        Ok(Self {
            normalized: bf16(hidden)?,
            packed_qkv: bf16(packed)?,
            raw_query: bf16(query)?,
            raw_key: bf16(kv)?,
            raw_value: bf16(kv)?,
            query: bf16(query)?,
            key: bf16(kv)?,
            value: bf16(kv)?,
            attended: bf16(query)?,
            projected: bf16(hidden)?,
            biased: bf16(hidden)?,
            residual: bf16(hidden)?,
            router: bf16(product(tokens, config.experts)?)?,
            router_biased: bf16(product(tokens, config.experts)?)?,
            selected: backend.inner.pool.allocate(&backend.inner.stream, routes)?,
            routing: bf16(routes)?,
            activated: bf16(product(routes, config.intermediate)?)?,
            moe: bf16(hidden)?,
        })
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("clamped-routed scratch size overflow"))
}
