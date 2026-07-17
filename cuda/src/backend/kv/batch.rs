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
        self.block_counts.host.fill(0);
        for (sequence, table) in batch.iter().copied().enumerate() {
            self.validate_table(table)?;
            let offset = sequence * self.max_blocks;
            for (target, block) in self.tables.host[offset..].iter_mut().zip(table.blocks()) {
                *target = block.0;
            }
            self.token_counts.host[sequence] = u32::try_from(table.token_len())?;
            self.block_counts.host[sequence] = u32::try_from(table.blocks().len())?;
        }
        self.tables.upload(&self.stream)?;
        self.token_counts.upload(&self.stream)?;
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

    pub(crate) const fn block_counts(&self) -> &DeviceBuffer<u32> {
        &self.block_counts.device
    }

    pub(crate) const fn max_blocks(&self) -> usize {
        self.max_blocks
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
