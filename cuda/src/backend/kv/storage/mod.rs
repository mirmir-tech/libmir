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
///
/// A storage without values (`value_head_dim == 0`) is a latent cache: one
/// row per token whose leading columns are also the token's value, so the
/// value pages alias the key pages and only latent attention writes them.
#[derive(Clone, Debug)]
pub struct PagedKvCache {
    /// Absent for a latent cache, which K/V stores never target.
    operation: Option<PagedKvStore>,
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
        if storage.value_head_dim == 0 {
            return Self::latent(backend, layer, storage, ring, block_count);
        }
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
            operation: Some(operation),
            stream: backend.inner.stream.clone(),
            storage,
            ring,
            layer,
            key_pages,
            value_pages,
        })
    }

    fn latent(
        backend: &CudaBackend,
        layer: usize,
        storage: KvStorageSpec,
        ring: Option<RingGeometry>,
        block_count: usize,
    ) -> Result<Self> {
        if ring.is_some()
            || storage.kv_heads != 1
            || storage.key_head_dim == 0
            || !matches!(storage.cache.dtype, KvCacheDType::Auto | KvCacheDType::BFloat16)
        {
            return Err(Error::InvalidPagedKv(
                "a latent cache is one unwindowed BF16 row per token",
            ));
        }
        let bytes = block_count
            .checked_mul(storage.cache.block_size)
            .and_then(|rows| rows.checked_mul(storage.key_head_dim))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or(Error::InvalidPagedKv("latent cache size overflow"))?;
        let rows = backend.inner.pool.allocate_zeroed::<u8>(&backend.inner.stream, bytes)?;
        Ok(Self {
            operation: None,
            stream: backend.inner.stream.clone(),
            storage,
            ring,
            layer,
            value_pages: rows.clone(),
            key_pages: rows,
        })
    }

    /// Whether the pages hold one latent row per token instead of keys and
    /// values.
    pub(crate) const fn is_latent(&self) -> bool {
        self.operation.is_none()
    }

    /// The latent rows of a latent cache.
    pub(crate) fn latent_rows_mut(&mut self) -> Result<&mut DeviceBuffer<u8>> {
        if self.is_latent() {
            Ok(&mut self.key_pages)
        } else {
            Err(Error::InvalidPagedKv("the cache holds keys and values, not latent rows"))
        }
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

    /// Device bytes held by the key and value pages of this layer.
    pub(crate) fn bytes(&self) -> usize {
        if self.is_latent() {
            return self.key_pages.bytes();
        }
        self.key_pages.bytes().saturating_add(self.value_pages.bytes())
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

    pub(crate) fn kernel(&self) -> Result<TypedKernel<KvStoreKernel>> {
        key_value(self.operation.as_ref()).map(PagedKvStore::kernel)
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
        let operation = key_value(self.operation.as_ref())?;
        for write in plan.writes() {
            operation.execute(
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

/// The K/V store of a cache with values; a latent cache has none.
fn key_value(operation: Option<&PagedKvStore>) -> Result<&PagedKvStore> {
    operation.ok_or(Error::InvalidPagedKv("a latent cache takes no K/V stores"))
}
