use mircuda::{Context, DeviceBuffer, FmhaBf16Plan, FmhaBf16Spec, Stream, bf16};
use runtime::kv::{KvCacheDType, KvStorageSpec};

use super::super::super::{PagedDecodeBatch, PagedKvCache};
use crate::{CudaBackend, Result};

#[derive(Debug)]
pub(super) struct PagedFmhaDecode {
    plan: FmhaBf16Plan,
    softmax_lse: DeviceBuffer<f32>,
    output_accum: DeviceBuffer<f32>,
    softmax_lse_accum: DeviceBuffer<f32>,
    stream: Stream,
    context: Context,
    max_blocks: usize,
    block_size: usize,
    query_heads: usize,
    head_dim: usize,
    tuned_band: usize,
    num_splits: usize,
}

const MAX_SPLITS: usize = 32;

impl PagedFmhaDecode {
    pub(super) fn prepare(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<Option<Self>> {
        let supported = matches!(storage.cache.dtype, KvCacheDType::Auto | KvCacheDType::BFloat16)
            && matches!(storage.key_head_dim, 64 | 128 | 256)
            && storage.value_head_dim == storage.key_head_dim;
        if !supported {
            return Ok(None);
        }
        Ok(Some(Self {
            plan: FmhaBf16Plan::new(
                &backend.inner.context,
                &backend.inner.stream,
                FmhaBf16Spec::new(
                    query_heads,
                    storage.kv_heads,
                    storage.key_head_dim,
                    storage.value_head_dim,
                )?,
            )?,
            softmax_lse: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, max_batch * query_heads)?,
            output_accum: backend.inner.pool.allocate(
                &backend.inner.stream,
                max_batch * query_heads * storage.key_head_dim * MAX_SPLITS,
            )?,
            softmax_lse_accum: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, max_batch * query_heads * MAX_SPLITS)?,
            stream: backend.inner.stream.clone(),
            context: backend.inner.context.clone(),
            max_blocks,
            block_size: storage.cache.block_size,
            query_heads,
            head_dim: storage.key_head_dim,
            tuned_band: 0,
            num_splits: 1,
        }))
    }

    pub(super) fn execute(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<()> {
        let band = context_band(batch.maximum_tokens());
        if self.tuned_band != band {
            self.tune(query, cache, batch, output, scale)?;
            self.tuned_band = band;
        }
        self.execute_with_splits(query, cache, batch, output, scale, self.num_splits)
    }

    fn execute_with_splits(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
        num_splits: usize,
    ) -> Result<()> {
        let rows = batch.active();
        if num_splits == 1 {
            Ok(self.plan.execute_paged_varlen(
                &self.stream,
                query,
                cache.key_pages(),
                cache.value_pages(),
                output,
                batch.query_starts(),
                batch.token_counts(),
                batch.context_starts(),
                batch.tables(),
                &mut self.softmax_lse,
                rows,
                rows,
                1,
                batch.maximum_tokens(),
                self.max_blocks,
                self.block_size,
                scale,
            )?)
        } else {
            Ok(self.plan.execute_paged_varlen_split(
                &self.stream,
                query,
                cache.key_pages(),
                cache.value_pages(),
                output,
                batch.query_starts(),
                batch.token_counts(),
                batch.context_starts(),
                batch.tables(),
                &mut self.softmax_lse,
                &mut self.output_accum,
                &mut self.softmax_lse_accum,
                num_splits,
                rows,
                rows,
                1,
                batch.maximum_tokens(),
                self.max_blocks,
                self.block_size,
                scale,
            )?)
        }
    }

    fn tune(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<()> {
        let mut selected = 1;
        let mut selected_ms = f32::INFINITY;
        for candidate in [1, 2, 4, 8, 16, 32] {
            self.execute_with_splits(query, cache, batch, output, scale, candidate)?;
            let started = self.context.create_event(true)?;
            let completed = self.context.create_event(true)?;
            started.record(&self.stream)?;
            for _ in 0..3 {
                self.execute_with_splits(query, cache, batch, output, scale, candidate)?;
            }
            completed.record(&self.stream)?;
            completed.synchronize()?;
            let average_ms = started.elapsed_ms(&completed)? / 3.0;
            if average_ms < selected_ms {
                selected = candidate;
                selected_ms = average_ms;
            }
        }
        self.num_splits = selected;
        tracing::info!(
            target: "libmir::cuda::tuning",
            rows = batch.active(),
            context_tokens = batch.maximum_tokens(),
            query_heads = self.query_heads,
            head_dim = self.head_dim,
            num_splits = selected,
            average_us = selected_ms * 1_000.0,
            "selected CUDA paged FlashAttention split count"
        );
        Ok(())
    }

    pub(super) fn capture_key(batch: &PagedDecodeBatch) -> usize {
        context_band(batch.maximum_tokens())
    }
}

fn context_band(tokens: usize) -> usize {
    tokens.max(1).next_power_of_two()
}
