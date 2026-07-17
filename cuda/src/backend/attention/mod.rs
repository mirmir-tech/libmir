use ::runtime::kv::{BlockTable, KvBackendStorage, KvWritePlan};
use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    Bf16LinearPack, Bf16LinearPackWeights, Bf16Projection, CudaBackend, NvFp4LinearWeight,
    PagedAttentionBf16, PagedKvCache, ProjectionFormat, RmsNormBf16,
};
use crate::{
    CudaTensor, Error, Result,
    kernels::{QkvPostprocess, QkvPostprocessSpec},
};

mod batch;
#[cfg(test)]
mod batch_tests;
mod config;
pub(in crate::backend) mod graph;
mod output;
mod prefill;
mod qkv;
#[cfg(test)]
mod tests;

pub use batch::BatchedDecodeAttentionBf16;
use config::validate;
pub use config::{DecodeAttentionConfig, DecodeAttentionWeights, DecodeQkvWeights};
pub use graph::CapturedDecodeAttentionBf16;
use output::AttentionOutputProjection;
pub use output::DecodeAttentionOutputWeight;
pub use prefill::PrefillAttentionBf16;
use qkv::{AttentionQkvProjection, QkvProjectionBuffers};

/// Allocation-free BF16 decode attention over runtime-owned paged K/V state.
#[derive(Debug)]
pub struct DecodeAttentionBf16 {
    input_norm: RmsNormBf16,
    qkv: AttentionQkvProjection,
    qkv_postprocess: QkvPostprocess,
    pub(in crate::backend) attention: PagedAttentionBf16,
    output_projection: AttentionOutputProjection,
    pub(in crate::backend) cache: PagedKvCache,
    pub(in crate::backend) scratch: AttentionScratch,
    stream: Stream,
    pub(in crate::backend) config: DecodeAttentionConfig,
}

#[derive(Debug)]
pub(in crate::backend) struct AttentionScratch {
    pub(in crate::backend) normalized: DeviceBuffer<bf16>,
    pub(in crate::backend) qkv: DeviceBuffer<bf16>,
    pub(in crate::backend) qkv_separate: [DeviceBuffer<bf16>; 3],
    pub(in crate::backend) value_norm: DeviceBuffer<bf16>,
    pub(in crate::backend) query_rope: DeviceBuffer<bf16>,
    pub(in crate::backend) key_rope: DeviceBuffer<bf16>,
    pub(in crate::backend) attention: DeviceBuffer<bf16>,
}

impl CudaBackend {
    /// Prepares the complete device-resident attention path for one decode
    /// token.
    pub fn prepare_decode_attention_bf16(
        &self,
        config: DecodeAttentionConfig,
    ) -> Result<DecodeAttentionBf16> {
        DecodeAttentionBf16::new(self, config)
    }
}

impl DecodeAttentionBf16 {
    fn new(backend: &CudaBackend, config: DecodeAttentionConfig) -> Result<Self> {
        let cache = backend.prepare_paged_kv(config.layer, config.cache)?;
        Self::new_with_cache(backend, config, cache)
    }

    pub(in crate::backend) fn new_with_cache(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        cache: PagedKvCache,
    ) -> Result<Self> {
        Self::new_with_cache_and_weights(backend, config, cache, None)
    }

    pub(in crate::backend) fn new_with_cache_and_weights(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        cache: PagedKvCache,
        weights: Option<DecodeAttentionWeights<'_>>,
    ) -> Result<Self> {
        validate(config)?;
        if cache.layer() != config.layer || cache.storage_spec() != config.cache {
            return Err(Error::InvalidPagedKv("shared cache differs from attention layer"));
        }
        let attention = backend.prepare_paged_attention_bf16(
            &cache,
            config.query_heads,
            config.max_sequence_blocks,
        )?;
        let qkv_postprocess = QkvPostprocessSpec {
            tokens: 1,
            query_heads: config.query_heads,
            kv_heads: config.cache.kv_heads,
            head_dim: config.cache.key_head_dim,
            value_head_dim: config.cache.value_head_dim,
            rotary_dim: config.rotary_dim,
            pairing_dim: config.rope_pairing_dim,
            theta: config.rope_theta,
            epsilon: config.rms_norm_epsilon,
            normalization: config.qkv_normalization,
        };
        let output_projection =
            AttentionOutputProjection::new(backend, config, 1, weights.map(|value| value.output))?;
        Ok(Self {
            input_norm: RmsNormBf16::new(backend, 1, config.hidden_size, config.rms_norm_epsilon)?,
            qkv: AttentionQkvProjection::new(backend, config, 1, weights.map(|value| value.qkv))?,
            qkv_postprocess: QkvPostprocess::compile(&backend.inner.compiler, qkv_postprocess)?,
            output_projection,
            scratch: AttentionScratch::new(backend, config)?,
            stream: backend.inner.stream.clone(),
            attention,
            cache,
            config,
        })
    }

    /// Enqueues a complete one-token attention pass without allocation or
    /// synchronization.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if write_plan.token_count() != 1 || table.token_len() == 0 {
            return Err(Error::InvalidPagedKv("decode attention requires exactly one new token"));
        }
        let position = table.token_len() - 1;
        let separate = self.qkv.execute(
            input,
            &self.input_norm,
            weights.input_norm,
            weights.qkv,
            &mut QkvProjectionBuffers {
                normalized: &mut self.scratch.normalized,
                packed: &mut self.scratch.qkv,
                separate: &mut self.scratch.qkv_separate,
            },
        )?;
        if separate {
            self.qkv_postprocess.execute_separate(
                &self.stream,
                [
                    &self.scratch.qkv_separate[0],
                    &self.scratch.qkv_separate[1],
                    &self.scratch.qkv_separate[2],
                ],
                weights.query_norm,
                weights.key_norm,
                &mut self.scratch.query_rope,
                &mut self.scratch.key_rope,
                &mut self.scratch.value_norm,
                position,
            )?;
        } else {
            self.qkv_postprocess.execute(
                &self.stream,
                &self.scratch.qkv,
                weights.query_norm,
                weights.key_norm,
                &mut self.scratch.query_rope,
                &mut self.scratch.key_rope,
                &mut self.scratch.value_norm,
                position,
            )?;
        }
        self.cache.store(write_plan, &self.scratch.key_rope, &self.scratch.value_norm)?;
        self.attention.execute(
            &self.scratch.query_rope,
            &self.cache,
            table,
            &mut self.scratch.attention,
            self.config.sliding_window,
            self.config.attention_scale,
        )?;
        self.output_projection.execute(
            &self.stream,
            &self.scratch.attention,
            weights.output,
            output,
        )
    }
}

impl AttentionScratch {
    fn new(backend: &CudaBackend, config: DecodeAttentionConfig) -> Result<Self> {
        let allocate =
            |elements| backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
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
        })
    }
}
