use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::{KvCacheDType, KvStorageSpec};

use super::PagedPrefillBatch;
use crate::{
    CudaBackend, Error, Result,
    kernels::{WindowedPrefillStage, WindowedPrefillStageArgs},
};

mod metadata;
use metadata::WindowedMetadata;

#[derive(Debug)]
pub struct WindowedPrefillStaging {
    stage: WindowedPrefillStage,
    stream: Stream,
    key_pages: DeviceBuffer<u8>,
    value_pages: DeviceBuffer<u8>,
    tables: WindowedMetadata,
    source_starts: WindowedMetadata,
    history_tokens: WindowedMetadata,
    context_tokens: WindowedMetadata,
    context_starts: WindowedMetadata,
    row_capacity: usize,
    query_capacity: usize,
    context_capacity: usize,
    blocks_per_row: usize,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    window_capacity: usize,
    max_context_tokens: usize,
}

impl WindowedPrefillStaging {
    pub(crate) fn new(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        row_capacity: usize,
        query_capacity: usize,
        window: usize,
    ) -> Result<Self> {
        validate(storage, row_capacity, query_capacity, window)?;
        let context_capacity = query_capacity
            .checked_add(window - 1)
            .ok_or(Error::InvalidPagedKv("windowed prefill context capacity overflow"))?;
        let blocks_per_row = context_capacity.div_ceil(storage.cache.block_size);
        let page_tokens = row_capacity
            .checked_mul(blocks_per_row)
            .and_then(|value| value.checked_mul(storage.cache.block_size))
            .ok_or(Error::InvalidPagedKv("windowed prefill page capacity overflow"))?;
        let page_bytes = page_tokens
            .checked_mul(storage.kv_heads)
            .and_then(|value| value.checked_mul(storage.key_head_dim))
            .and_then(|value| value.checked_mul(size_of::<bf16>()))
            .ok_or(Error::InvalidPagedKv("windowed prefill page bytes overflow"))?;
        let mut tables = WindowedMetadata::new(backend, row_capacity * blocks_per_row, u32::MAX)?;
        for row in 0..row_capacity {
            for block in 0..blocks_per_row {
                tables.host[row * blocks_per_row + block] =
                    u32::try_from(row * blocks_per_row + block)?;
            }
        }
        tables.upload(&backend.inner.stream)?;
        Ok(Self {
            stage: WindowedPrefillStage::compile(backend)?,
            stream: backend.inner.stream.clone(),
            key_pages: backend.inner.pool.allocate(&backend.inner.stream, page_bytes)?,
            value_pages: backend.inner.pool.allocate(&backend.inner.stream, page_bytes)?,
            tables,
            source_starts: WindowedMetadata::new(backend, row_capacity, 0)?,
            history_tokens: WindowedMetadata::new(backend, row_capacity, 0)?,
            context_tokens: WindowedMetadata::new(backend, row_capacity, 0)?,
            context_starts: WindowedMetadata::new(backend, row_capacity + 1, 0)?,
            row_capacity,
            query_capacity,
            context_capacity,
            blocks_per_row,
            block_size: storage.cache.block_size,
            kv_heads: storage.kv_heads,
            head_dim: storage.key_head_dim,
            window_capacity: window,
            max_context_tokens: 0,
        })
    }

    pub(crate) fn matches(
        &self,
        rows: usize,
        queries: usize,
        storage: KvStorageSpec,
        window: usize,
    ) -> bool {
        self.row_capacity >= rows
            && self.query_capacity >= queries
            && self.block_size == storage.cache.block_size
            && self.kv_heads == storage.kv_heads
            && self.head_dim == storage.key_head_dim
            && self.window_capacity >= window
    }

    pub(crate) fn stage(
        &mut self,
        batch: &PagedPrefillBatch,
        current_keys: &DeviceBuffer<bf16>,
        current_values: &DeviceBuffer<bf16>,
        ring_keys: &DeviceBuffer<u8>,
        ring_values: &DeviceBuffer<u8>,
        window: usize,
    ) -> Result<()> {
        self.prepare(batch, window)?;
        self.stage.execute(
            &self.stream,
            &mut WindowedPrefillStageArgs {
                current_keys,
                current_values,
                ring_keys,
                ring_values,
                staged_keys: &mut self.key_pages,
                staged_values: &mut self.value_pages,
                ring_tables: batch.ring_tables(),
                query_starts: batch.query_starts(),
                source_starts: &self.source_starts.device,
                history_tokens: &self.history_tokens.device,
                context_tokens: &self.context_tokens.device,
                active_rows: batch.active(),
                max_context_tokens: self.max_context_tokens,
                ring_max_blocks: batch.max_blocks(),
                staged_blocks_per_row: self.blocks_per_row,
                block_size: self.block_size,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
            },
        )
    }

    fn prepare(&mut self, batch: &PagedPrefillBatch, window: usize) -> Result<()> {
        if batch.active() > self.row_capacity
            || batch.tokens() > self.query_capacity
            || window == 0
            || window > self.window_capacity
        {
            return Err(Error::InvalidPagedKv("windowed prefill staging capacity exceeded"));
        }
        self.source_starts.host.fill(0);
        self.history_tokens.host.fill(0);
        self.context_tokens.host.fill(0);
        self.context_starts.host.fill(0);
        let mut packed_context = 0_usize;
        self.max_context_tokens = 0;
        for (row, item) in batch.rows().iter().enumerate() {
            let history = item.start().min(window - 1);
            let context = history
                .checked_add(item.tokens())
                .ok_or(Error::InvalidPagedKv("windowed prefill row context overflow"))?;
            self.source_starts.host[row] = u32::try_from(item.start() - history)?;
            self.history_tokens.host[row] = u32::try_from(history)?;
            self.context_tokens.host[row] = u32::try_from(context)?;
            packed_context = packed_context
                .checked_add(context)
                .ok_or(Error::InvalidPagedKv("windowed prefill packed context overflow"))?;
            self.context_starts.host[row + 1] = u32::try_from(packed_context)?;
            self.max_context_tokens = self.max_context_tokens.max(context);
        }
        if self.max_context_tokens == 0 || self.max_context_tokens > self.context_capacity {
            return Err(Error::InvalidPagedKv("invalid windowed prefill staged context"));
        }
        self.source_starts.upload(&self.stream)?;
        self.history_tokens.upload(&self.stream)?;
        self.context_tokens.upload(&self.stream)?;
        self.context_starts.upload(&self.stream)
    }

    pub(crate) const fn key_pages(&self) -> &DeviceBuffer<u8> {
        &self.key_pages
    }

    pub(crate) const fn value_pages(&self) -> &DeviceBuffer<u8> {
        &self.value_pages
    }

    pub(crate) const fn tables(&self) -> &DeviceBuffer<u32> {
        &self.tables.device
    }

    pub(crate) const fn token_counts(&self) -> &DeviceBuffer<u32> {
        &self.context_tokens.device
    }

    pub(crate) const fn context_starts(&self) -> &DeviceBuffer<u32> {
        &self.context_starts.device
    }

    pub(crate) fn fmha_max_context_tokens(&self) -> usize {
        self.max_context_tokens
            .div_ceil(128)
            .saturating_mul(128)
            .min(self.blocks_per_row * self.block_size)
    }

    pub(crate) const fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }
}

fn validate(storage: KvStorageSpec, rows: usize, queries: usize, window: usize) -> Result<()> {
    if rows == 0
        || queries == 0
        || window == 0
        || storage.cache.block_size == 0
        || storage.key_head_dim != storage.value_head_dim
        || !matches!(storage.cache.dtype, KvCacheDType::Auto | KvCacheDType::BFloat16)
    {
        return Err(Error::InvalidPagedKv("invalid windowed prefill staging geometry"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
