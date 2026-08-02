use ::runtime::kv::{KvBackendStorage, KvCacheDType, KvCacheLayout, KvStorageSpec, KvWritePlan};
use mircuda::{DeviceBuffer, Stream, TypedKernel, bf16};

use super::super::CudaBackend;
use crate::{
    Error, Result,
    kernels::{KvStoreKernel, PagedKvSpec, PagedKvStore},
};

mod operations;
mod ring;
use ring::RingGeometry;

/// Device-resident encoded K/V pages addressed by runtime-owned physical
/// blocks.
#[derive(Clone, Debug)]
pub struct PagedKvCache {
    operation: PagedKvStore,
    stream: Stream,
    storage: KvStorageSpec,
    ring: Option<RingGeometry>,
    layer: usize,
    key_pages: DeviceBuffer<u8>,
    value_pages: DeviceBuffer<u8>,
}

impl PagedKvCache {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        layer: usize,
        storage: KvStorageSpec,
    ) -> Result<Self> {
        Self::with_ring(backend, layer, storage, None)
    }

    pub(in crate::backend) fn new_windowed(
        backend: &CudaBackend,
        layer: usize,
        storage: KvStorageSpec,
        window: usize,
        sessions: usize,
    ) -> Result<Self> {
        let ring = RingGeometry::new(window, storage.cache, sessions)?;
        Self::with_ring(backend, layer, storage, Some(ring))
    }

    fn with_ring(
        backend: &CudaBackend,
        layer: usize,
        storage: KvStorageSpec,
        ring: Option<RingGeometry>,
    ) -> Result<Self> {
        if storage.native_bits != 16 {
            return Err(Error::InvalidPagedKv("CUDA pages require 16 native activation bits"));
        }
        if storage.layout != KvCacheLayout::Nhd {
            return Err(Error::InvalidPagedKv("CUDA paged BF16 storage requires NHD layout"));
        }
        let block_count = ring.map_or_else(
            || Ok(usize::try_from(storage.cache.block_count)?),
            RingGeometry::physical_blocks,
        )?;
        let spec = PagedKvSpec {
            block_size: storage.cache.block_size,
            block_count,
            kv_heads: storage.kv_heads,
            key_head_dim: storage.key_head_dim,
            value_head_dim: storage.value_head_dim,
            dtype: storage.cache.dtype,
        };
        let operation = PagedKvStore::compile(&backend.inner.compiler, spec)?;
        let key_pages = backend
            .inner
            .pool
            .allocate_zeroed::<u8>(&backend.inner.stream, operation.key_bytes()?)?;
        let value_pages = backend
            .inner
            .pool
            .allocate_zeroed::<u8>(&backend.inner.stream, operation.value_bytes()?)?;
        Ok(Self {
            operation,
            stream: backend.inner.stream.clone(),
            storage,
            ring,
            layer,
            key_pages,
            value_pages,
        })
    }

    pub(crate) const fn is_windowed(&self) -> bool {
        self.ring.is_some()
    }

    #[must_use]
    pub const fn storage_spec(&self) -> KvStorageSpec {
        self.storage
    }

    pub(crate) const fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) const fn key_pages(&self) -> &DeviceBuffer<u8> {
        &self.key_pages
    }

    pub(crate) const fn value_pages(&self) -> &DeviceBuffer<u8> {
        &self.value_pages
    }

    pub(crate) fn pages_mut(&mut self) -> (&mut DeviceBuffer<u8>, &mut DeviceBuffer<u8>) {
        (&mut self.key_pages, &mut self.value_pages)
    }

    pub(crate) fn kernel(&self) -> TypedKernel<KvStoreKernel> {
        self.operation.kernel()
    }

    pub(super) fn validate_plan(&self, plan: &KvWritePlan) -> Result<()> {
        if plan.block_size() != self.storage.cache.block_size {
            return Err(Error::InvalidPagedKv("write plan uses another KV block size"));
        }
        if plan.writes().iter().any(|write| write.page.layer != self.layer) {
            return Err(Error::InvalidPagedKv("write plan targets another decoder layer"));
        }
        Ok(())
    }
}

impl KvBackendStorage for PagedKvCache {
    type Error = Error;
    type Tensor = DeviceBuffer<bf16>;

    fn dtype(&self) -> KvCacheDType {
        self.storage.cache.dtype
    }

    fn store(
        &mut self,
        plan: &KvWritePlan,
        keys: &Self::Tensor,
        values: &Self::Tensor,
    ) -> Result<usize> {
        self.validate_plan(plan)?;
        for write in plan.writes() {
            self.operation.execute(
                &self.stream,
                keys,
                values,
                &mut self.key_pages,
                &mut self.value_pages,
                write.local_start,
                write.token_count(),
                usize::try_from(write.page.block.0)?,
                write.page_start,
            )?;
        }
        Ok(plan.written_tokens())
    }

    fn resident_token_slots(&self) -> usize {
        self.ring
            .and_then(|ring| ring.physical_blocks().ok())
            .unwrap_or_else(|| {
                usize::try_from(self.storage.cache.block_count).unwrap_or(usize::MAX)
            })
            .saturating_mul(self.storage.cache.block_size)
    }
}
