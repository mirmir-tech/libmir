use mircuda::{DeviceBuffer, PinnedBuffer, Stream};
use runtime::kv::{BlockTable, CacheConfig, KvStorageSpec};

use crate::{CudaBackend, Error, Result};

#[derive(Debug)]
struct U32Metadata {
    host: Vec<u32>,
    staging: PinnedBuffer<u32>,
    device: DeviceBuffer<u32>,
}

impl U32Metadata {
    fn new(backend: &CudaBackend, len: usize, fill: u32) -> Result<Self> {
        Ok(Self {
            host: vec![fill; len],
            staging: backend.inner.context.allocate_pinned(len)?,
            device: backend.inner.pool.allocate(&backend.inner.stream, len)?,
        })
    }

    fn upload(&mut self, stream: &Stream) -> Result<()> {
        self.staging.copy_from_slice(&self.host)?;
        Ok(stream.copy_to_device(&mut self.staging, &mut self.device)?)
    }
}

/// Shared device metadata for one fixed-capacity decode microbatch.
#[derive(Debug)]
pub struct PagedDecodeBatch {
    tables: U32Metadata,
    token_counts: U32Metadata,
    query_starts: U32Metadata,
    context_starts: U32Metadata,
    positions: U32Metadata,
    block_counts: U32Metadata,
    stream: Stream,
    cache: CacheConfig,
    max_batch: usize,
    max_blocks: usize,
    active: usize,
}

impl PagedDecodeBatch {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<Self> {
        if max_batch == 0 || max_blocks == 0 {
            return Err(Error::InvalidPagedKv("paged decode batch capacity is empty"));
        }
        let table_len = max_batch
            .checked_mul(max_blocks)
            .ok_or(Error::InvalidPagedKv("batched block table capacity overflow"))?;
        Ok(Self {
            tables: U32Metadata::new(backend, table_len, u32::MAX)?,
            token_counts: U32Metadata::new(backend, max_batch, 0)?,
            query_starts: U32Metadata::new(backend, max_batch + 1, 0)?,
            context_starts: U32Metadata::new(backend, max_batch + 1, 0)?,
            positions: U32Metadata::new(backend, max_batch, 0)?,
            block_counts: U32Metadata::new(backend, max_batch, 0)?,
            stream: backend.inner.stream.clone(),
            cache: storage.cache,
            max_batch,
            max_blocks,
            active: 0,
        })
    }

    /// Rebinds logical sequences without allocating or synchronizing the host.
    pub fn prepare(&mut self, batch: &[&BlockTable]) -> Result<()> {
        if batch.is_empty() || batch.len() > self.max_batch {
            return Err(Error::InvalidPagedKv("invalid block table batch size"));
        }
        self.tables.host.fill(u32::MAX);
        self.token_counts.host.fill(0);
        self.query_starts.host.fill(0);
        self.context_starts.host.fill(0);
        self.positions.host.fill(0);
        self.block_counts.host.fill(0);
        let mut packed_context = 0_u32;
        for (sequence, table) in batch.iter().copied().enumerate() {
            self.validate_table(table)?;
            let offset = sequence * self.max_blocks;
            for (target, block) in self.tables.host[offset..].iter_mut().zip(table.blocks()) {
                *target = block.0;
            }
            self.token_counts.host[sequence] = u32::try_from(table.token_len())?;
            self.query_starts.host[sequence + 1] = u32::try_from(sequence + 1)?;
            packed_context = packed_context
                .checked_add(u32::try_from(table.token_len())?)
                .ok_or(Error::InvalidPagedKv("batched context offsets overflow"))?;
            self.context_starts.host[sequence + 1] = packed_context;
            self.positions.host[sequence] = u32::try_from(table.token_len() - 1)?;
            self.block_counts.host[sequence] = u32::try_from(table.blocks().len())?;
        }
        self.tables.upload(&self.stream)?;
        self.token_counts.upload(&self.stream)?;
        self.query_starts.upload(&self.stream)?;
        self.context_starts.upload(&self.stream)?;
        self.positions.upload(&self.stream)?;
        self.block_counts.upload(&self.stream)?;
        self.active = batch.len();
        Ok(())
    }

    #[must_use]
    pub const fn active(&self) -> usize {
        self.active
    }

    pub(crate) const fn tables(&self) -> &DeviceBuffer<u32> {
        &self.tables.device
    }

    pub(crate) const fn token_counts(&self) -> &DeviceBuffer<u32> {
        &self.token_counts.device
    }

    pub(crate) const fn query_starts(&self) -> &DeviceBuffer<u32> {
        &self.query_starts.device
    }

    pub(crate) const fn context_starts(&self) -> &DeviceBuffer<u32> {
        &self.context_starts.device
    }

    pub(crate) const fn positions(&self) -> &DeviceBuffer<u32> {
        &self.positions.device
    }

    pub(crate) const fn block_counts(&self) -> &DeviceBuffer<u32> {
        &self.block_counts.device
    }

    pub(crate) const fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    pub(crate) fn maximum_tokens(&self) -> usize {
        self.token_counts.host[..self.active]
            .iter()
            .copied()
            .max()
            .map_or(0, |tokens| tokens as usize)
    }

    pub(crate) fn tuning_sample(&self, backend: &CudaBackend, tokens: usize) -> Result<Self> {
        if tokens == 0 || self.active == 0 {
            return Err(Error::InvalidPagedKv("empty paged decode tuning sample"));
        }
        let table_len = self
            .max_batch
            .checked_mul(self.max_blocks)
            .ok_or(Error::InvalidPagedKv("batched block table capacity overflow"))?;
        let mut sample = Self {
            tables: U32Metadata::new(backend, table_len, u32::MAX)?,
            token_counts: U32Metadata::new(backend, self.max_batch, 0)?,
            query_starts: U32Metadata::new(backend, self.max_batch + 1, 0)?,
            context_starts: U32Metadata::new(backend, self.max_batch + 1, 0)?,
            positions: U32Metadata::new(backend, self.max_batch, 0)?,
            block_counts: U32Metadata::new(backend, self.max_batch, 0)?,
            stream: backend.inner.stream.clone(),
            cache: self.cache,
            max_batch: self.max_batch,
            max_blocks: self.max_blocks,
            active: self.active,
        };
        sample.tables.host.copy_from_slice(&self.tables.host);
        sample.query_starts.host.copy_from_slice(&self.query_starts.host);
        for row in 0..self.active {
            let visible = tokens.min(self.token_counts.host[row] as usize).max(1);
            sample.token_counts.host[row] = u32::try_from(visible)?;
            sample.positions.host[row] = u32::try_from(visible - 1)?;
            sample.block_counts.host[row] = u32::try_from(visible.div_ceil(self.cache.block_size))?;
            sample.context_starts.host[row + 1] = sample.context_starts.host[row]
                .checked_add(u32::try_from(visible)?)
                .ok_or(Error::InvalidPagedKv("sample context offsets overflow"))?;
        }
        sample.tables.upload(&sample.stream)?;
        sample.token_counts.upload(&sample.stream)?;
        sample.query_starts.upload(&sample.stream)?;
        sample.context_starts.upload(&sample.stream)?;
        sample.positions.upload(&sample.stream)?;
        sample.block_counts.upload(&sample.stream)?;
        Ok(sample)
    }

    pub(crate) const fn cache_config(&self) -> CacheConfig {
        self.cache
    }

    fn validate_table(&self, table: &BlockTable) -> Result<()> {
        if table.block_size() != Some(self.cache.block_size)
            || table.blocks().is_empty()
            || table.blocks().len() > self.max_blocks
        {
            return Err(Error::InvalidPagedKv("invalid batched block table geometry"));
        }
        let capacity = table
            .blocks()
            .len()
            .checked_mul(self.cache.block_size)
            .ok_or(Error::InvalidPagedKv("batched block table capacity overflow"))?;
        if table.token_len() == 0 || table.token_len() > capacity {
            return Err(Error::InvalidPagedKv("invalid batched block table token count"));
        }
        let physical_blocks = self.cache.block_count;
        if table.blocks().iter().any(|block| block.0 >= physical_blocks) {
            return Err(Error::InvalidPagedKv("batched block table references a missing page"));
        }
        Ok(())
    }
}
