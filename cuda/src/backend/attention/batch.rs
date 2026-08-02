use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    super::{BatchedPagedAttentionBf16, PagedDecodeBatch, PagedKvCache},
    AttentionOutputProjection, AttentionQkvProjection, DecodeAttentionConfig,
    DecodeAttentionWeights, QkvProjectionBuffers, validate,
};
use crate::{
    CudaBackend, Error, Result, RmsNormBf16,
    kernels::{BatchedQkvPostprocess, BatchedSplitAttentionWorkspace, QkvPostprocessSpec},
};

/// Fixed-shape attention for independent one-token decode rows.
#[derive(Debug)]
pub struct BatchedDecodeAttentionBf16 {
    input_norm: RmsNormBf16,
    qkv: AttentionQkvProjection,
    qkv_postprocess: BatchedQkvPostprocess,
    attention: BatchedPagedAttentionBf16,
    output_projection: AttentionOutputProjection,
    cache: PagedKvCache,
    scratch: BatchAttentionScratch,
    stream: Stream,
    config: DecodeAttentionConfig,
    batch_size: usize,
}

#[derive(Debug)]
struct BatchAttentionScratch {
    normalized: DeviceBuffer<bf16>,
    qkv: DeviceBuffer<bf16>,
    qkv_separate: [DeviceBuffer<bf16>; 3],
    value_norm: DeviceBuffer<bf16>,
    query_rope: DeviceBuffer<bf16>,
    key_rope: DeviceBuffer<bf16>,
    attention: DeviceBuffer<bf16>,
}

impl CudaBackend {
    /// Prepares one complete attention path for a fixed decode microbatch.
    pub fn prepare_batched_decode_attention_bf16(
        &self,
        config: DecodeAttentionConfig,
        batch_size: usize,
    ) -> Result<BatchedDecodeAttentionBf16> {
        BatchedDecodeAttentionBf16::new(self, config, batch_size)
    }
}

impl BatchedDecodeAttentionBf16 {
    fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        batch_size: usize,
    ) -> Result<Self> {
        let cache = backend.prepare_paged_kv(config.layer, config.cache)?;
        Self::new_with_cache(backend, config, batch_size, cache)
    }

    pub(in crate::backend) fn new_with_cache(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        batch_size: usize,
        cache: PagedKvCache,
    ) -> Result<Self> {
        Self::new_with_cache_and_weights(backend, config, batch_size, cache, None)
    }

    pub(in crate::backend) fn new_with_cache_and_weights(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        batch_size: usize,
        cache: PagedKvCache,
        weights: Option<DecodeAttentionWeights<'_>>,
    ) -> Result<Self> {
        Self::new_with_cache_weights_workspace(backend, config, batch_size, cache, weights, None)
    }

    pub(in crate::backend) fn new_with_cache_weights_workspace(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        batch_size: usize,
        cache: PagedKvCache,
        weights: Option<DecodeAttentionWeights<'_>>,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        validate(config)?;
        if batch_size == 0 {
            return Err(Error::InvalidPagedKv("decode attention batch is empty"));
        }
        if cache.layer() != config.layer || cache.storage_spec() != config.cache {
            return Err(Error::InvalidPagedKv("shared cache differs from batched attention"));
        }
        let attention = BatchedPagedAttentionBf16::new_with_workspace(
            backend,
            &cache,
            config.query_heads,
            config.max_sequence_blocks,
            batch_size,
            workspace,
        )?;
        let qkv_spec = QkvPostprocessSpec {
            tokens: batch_size,
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
        Ok(Self {
            input_norm: RmsNormBf16::new(
                backend,
                batch_size,
                config.hidden_size,
                config.rms_norm_epsilon,
            )?,
            qkv: AttentionQkvProjection::new(
                backend,
                config,
                batch_size,
                weights.map(|value| value.qkv),
            )?,
            qkv_postprocess: BatchedQkvPostprocess::compile(&backend.inner.compiler, qkv_spec)?,
            output_projection: AttentionOutputProjection::new(
                backend,
                config,
                batch_size,
                weights.map(|value| value.output),
            )?,
            scratch: BatchAttentionScratch::new(backend, config, batch_size)?,
            stream: backend.inner.stream.clone(),
            attention,
            cache,
            config,
            batch_size,
        })
    }

    /// Executes `RMSNorm`, QKV, per-row `RoPE`, K/V store, attention, and
    /// output.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if batch.active() != self.batch_size || batch.cache_config() != self.config.cache.cache {
            return Err(Error::InvalidPagedKv("decode attention batch differs from plan"));
        }
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
        self.qkv_postprocess.execute(
            &self.stream,
            if separate {
                [
                    &self.scratch.qkv_separate[0],
                    &self.scratch.qkv_separate[1],
                    &self.scratch.qkv_separate[2],
                ]
            } else {
                [&self.scratch.qkv, &self.scratch.qkv, &self.scratch.qkv]
            },
            separate,
            weights.query_norm,
            weights.key_norm,
            batch.positions(),
            &mut self.scratch.query_rope,
            &mut self.scratch.key_rope,
            &mut self.scratch.value_norm,
        )?;
        self.cache
            .store_batch(batch, &self.scratch.key_rope, &self.scratch.value_norm)?;
        self.attention.execute(
            &self.scratch.query_rope,
            &self.cache,
            batch,
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

impl BatchAttentionScratch {
    fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        batch_size: usize,
    ) -> Result<Self> {
        let allocate = |width: usize| -> Result<DeviceBuffer<bf16>> {
            let elements = batch_size
                .checked_mul(width)
                .ok_or(Error::InvalidPagedKv("decode attention batch scratch overflow"))?;
            Ok(backend.inner.pool.allocate(&backend.inner.stream, elements)?)
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
        })
    }
}
