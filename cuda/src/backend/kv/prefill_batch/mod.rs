use mircuda::{DeviceBuffer, Stream};
use runtime::kv::{BlockTable, CacheConfig, KvStorageSpec};

use crate::{CudaBackend, Error, Result};

mod capacity;
mod decode;
mod metadata;
mod ring;
mod row;

use metadata::Metadata;
pub use row::PrefillBatchRow;

/// Packed metadata for one variable-length prefill microbatch.
#[derive(Debug)]
pub struct PagedPrefillBatch {
    tables: Metadata,
    token_counts: Metadata,
    block_counts: Metadata,
    query_starts: Metadata,
    context_starts: Metadata,
    request_indices: Metadata,
    positions: Metadata,
    slot_mapping: Metadata,
    ring_tables: Metadata,
    ring_slot_mapping: Metadata,
    rows: Vec<PrefillBatchRow>,
    stream: Stream,
    cache: CacheConfig,
    max_batch: usize,
    max_blocks: usize,
    max_tokens: usize,
    active: usize,
    tokens: usize,
    max_query_tokens: usize,
    max_context_tokens: usize,
    decode_layout_rows: Option<usize>,
}

impl PagedPrefillBatch {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        max_blocks: usize,
        max_batch: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        if max_blocks == 0 || max_batch == 0 || max_tokens == 0 {
            return Err(Error::InvalidPagedKv("paged prefill batch capacity is empty"));
        }
        let table_len = max_batch
            .checked_mul(max_blocks)
            .ok_or(Error::InvalidPagedKv("prefill block table capacity overflow"))?;
        Ok(Self {
            tables: Metadata::new(backend, table_len, u32::MAX)?,
            token_counts: Metadata::new(backend, max_batch, 0)?,
            block_counts: Metadata::new(backend, max_batch, 0)?,
            query_starts: Metadata::new(backend, max_batch + 1, 0)?,
            context_starts: Metadata::new(backend, max_batch + 1, 0)?,
            request_indices: Metadata::new(backend, max_tokens, 0)?,
            positions: Metadata::new(backend, max_tokens, 0)?,
            slot_mapping: Metadata::new(backend, max_tokens, u32::MAX)?,
            ring_tables: Metadata::new(backend, table_len, u32::MAX)?,
            ring_slot_mapping: Metadata::new(backend, max_tokens, u32::MAX)?,
            rows: Vec::with_capacity(max_batch),
            stream: backend.inner.stream.clone(),
            cache: storage.cache,
            max_batch,
            max_blocks,
            max_tokens,
            active: 0,
            tokens: 0,
            max_query_tokens: 0,
            max_context_tokens: 0,
            decode_layout_rows: None,
        })
    }

    /// Packs block tables, absolute positions, and physical K/V slots.
    pub fn prepare(
        &mut self,
        tables: &[&BlockTable],
        starts: &[usize],
        query_tokens: &[usize],
    ) -> Result<()> {
        self.decode_layout_rows = None;
        self.validate_batch(tables, starts, query_tokens)?;
        self.clear();
        let mut packed = 0;
        let mut packed_context = 0;
        for (row, ((table, start), count)) in
            tables.iter().zip(starts).zip(query_tokens).enumerate()
        {
            self.rows.push(PrefillBatchRow::new((*table).clone(), *start, *count));
            self.prepare_row(row, table, *start, *count, packed)?;
            packed += count;
            packed_context += table.token_len();
            self.query_starts.host[row + 1] = u32::try_from(packed)?;
            self.context_starts.host[row + 1] = u32::try_from(packed_context)?;
            self.max_query_tokens = self.max_query_tokens.max(*count);
            self.max_context_tokens = self.max_context_tokens.max(table.token_len());
        }
        self.upload()?;
        self.active = tables.len();
        self.tokens = packed;
        Ok(())
    }

    fn prepare_row(
        &mut self,
        row: usize,
        table: &BlockTable,
        start: usize,
        count: usize,
        packed: usize,
    ) -> Result<()> {
        let table_offset = row * self.max_blocks;
        for (target, block) in self.tables.host[table_offset..].iter_mut().zip(table.blocks()) {
            *target = block.0;
        }
        self.token_counts.host[row] = u32::try_from(table.token_len())?;
        self.block_counts.host[row] = u32::try_from(table.blocks().len())?;
        for local in 0..count {
            let position = start + local;
            let block = table.blocks()[position / self.cache.block_size].0;
            let page_token = u64::from(block)
                .checked_mul(u64::try_from(self.cache.block_size)?)
                .and_then(|value| {
                    value.checked_add(u64::try_from(position % self.cache.block_size).ok()?)
                })
                .ok_or(Error::InvalidPagedKv("prefill slot mapping overflow"))?;
            self.positions.host[packed + local] = u32::try_from(position)?;
            self.request_indices.host[packed + local] = u32::try_from(row)?;
            self.slot_mapping.host[packed + local] = u32::try_from(page_token)?;
        }
        Ok(())
    }

    fn validate_batch(
        &self,
        tables: &[&BlockTable],
        starts: &[usize],
        query_tokens: &[usize],
    ) -> Result<()> {
        if tables.is_empty()
            || tables.len() > self.max_batch
            || starts.len() != tables.len()
            || query_tokens.len() != tables.len()
            || query_tokens.iter().sum::<usize>() > self.max_tokens
        {
            return Err(Error::InvalidPagedKv("invalid paged prefill batch geometry"));
        }
        for ((table, start), count) in tables.iter().zip(starts).zip(query_tokens) {
            let end = start
                .checked_add(*count)
                .ok_or(Error::InvalidPagedKv("prefill query range overflow"))?;
            let valid = *count > 0
                && table.block_size() == Some(self.cache.block_size)
                && !table.blocks().is_empty()
                && table.blocks().len() <= self.max_blocks
                && end == table.token_len()
                && table.blocks().iter().all(|block| block.0 < self.cache.block_count);
            if !valid {
                return Err(Error::InvalidPagedKv("invalid prefill block table row"));
            }
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.tables.host.fill(u32::MAX);
        self.token_counts.host.fill(0);
        self.block_counts.host.fill(0);
        self.query_starts.host.fill(0);
        self.context_starts.host.fill(0);
        self.request_indices.host.fill(0);
        self.positions.host.fill(0);
        self.slot_mapping.host.fill(u32::MAX);
        self.rows.clear();
        self.max_query_tokens = 0;
        self.max_context_tokens = 0;
    }

    fn upload(&mut self) -> Result<()> {
        self.tables.upload(&self.stream)?;
        self.token_counts.upload(&self.stream)?;
        self.block_counts.upload(&self.stream)?;
        self.query_starts.upload(&self.stream)?;
        self.context_starts.upload(&self.stream)?;
        self.request_indices.upload(&self.stream)?;
        self.positions.upload(&self.stream)?;
        self.slot_mapping.upload(&self.stream)
    }

    #[must_use]
    pub const fn active(&self) -> usize {
        self.active
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) const fn slot_mapping(&self, windowed: bool) -> &DeviceBuffer<u32> {
        if windowed {
            &self.ring_slot_mapping.device
        } else {
            &self.slot_mapping.device
        }
    }

    pub(crate) const fn tables(&self) -> &DeviceBuffer<u32> {
        &self.tables.device
    }

    pub(crate) const fn block_counts(&self) -> &DeviceBuffer<u32> {
        &self.block_counts.device
    }

    pub(crate) const fn query_starts(&self) -> &DeviceBuffer<u32> {
        &self.query_starts.device
    }

    pub(crate) const fn context_starts(&self) -> &DeviceBuffer<u32> {
        &self.context_starts.device
    }

    pub(crate) const fn request_indices(&self) -> &DeviceBuffer<u32> {
        &self.request_indices.device
    }

    pub(crate) const fn positions(&self) -> &DeviceBuffer<u32> {
        &self.positions.device
    }

    pub(crate) fn rows(&self) -> &[PrefillBatchRow] {
        &self.rows
    }

    pub(crate) const fn max_query_tokens(&self) -> usize {
        self.max_query_tokens
    }

    pub(crate) const fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    pub(crate) const fn cache_config(&self) -> CacheConfig {
        self.cache
    }
}
