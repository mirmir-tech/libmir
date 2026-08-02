use std::collections::HashMap;

use mircuda::{DeviceBuffer, MemoryPool, Stream, bf16};

use super::DecodeAttentionConfig;
use crate::{CudaBackend, Error, Result};

#[derive(Debug)]
pub(super) struct PrefillAttentionScratch {
    pub(super) normalized: DeviceBuffer<bf16>,
    pub(super) qkv: DeviceBuffer<bf16>,
    pub(super) qkv_separate: [DeviceBuffer<bf16>; 3],
    pub(super) value_norm: DeviceBuffer<bf16>,
    pub(super) query_rope: DeviceBuffer<bf16>,
    pub(super) key_rope: DeviceBuffer<bf16>,
    pub(super) attention: DeviceBuffer<bf16>,
    pub(super) rows: HashMap<usize, PrefillRowScratch>,
}

#[derive(Debug)]
pub(super) struct PrefillRowScratch {
    pub(super) query: DeviceBuffer<bf16>,
    pub(super) attention: DeviceBuffer<bf16>,
}

impl PrefillAttentionScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        tokens: usize,
    ) -> Result<Self> {
        let allocate = |width| -> Result<DeviceBuffer<bf16>> {
            let elements = tokens
                .checked_mul(width)
                .ok_or(Error::InvalidPagedKv("prefill attention scratch overflow"))?;
            Ok(backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements)?)
        };
        let query = config.query_heads * config.cache.key_head_dim;
        let key = config.cache.kv_heads * config.cache.key_head_dim;
        let value = config.cache.kv_heads * config.cache.value_head_dim;
        Ok(Self {
            normalized: allocate(config.hidden_size)?,
            qkv: allocate(query + key + value)?,
            qkv_separate: [allocate(query)?, allocate(key)?, allocate(value)?],
            value_norm: allocate(value)?,
            query_rope: allocate(query)?,
            key_rope: allocate(key)?,
            attention: allocate(config.query_heads * config.cache.value_head_dim)?,
            rows: HashMap::new(),
        })
    }

    pub(super) fn ensure_row(
        &mut self,
        pool: &MemoryPool,
        stream: &Stream,
        tokens: usize,
        query_width: usize,
        output_width: usize,
    ) -> Result<()> {
        if self.rows.contains_key(&tokens) {
            return Ok(());
        }
        let query = tokens
            .checked_mul(query_width)
            .ok_or(Error::InvalidPagedKv("prefill row query overflow"))?;
        let attention = tokens
            .checked_mul(output_width)
            .ok_or(Error::InvalidPagedKv("prefill row output overflow"))?;
        self.rows.insert(
            tokens,
            PrefillRowScratch {
                query: pool.allocate(stream, query)?,
                attention: pool.allocate(stream, attention)?,
            },
        );
        Ok(())
    }
}
