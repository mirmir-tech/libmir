use mircuda::{DeviceBuffer, FmhaBf16Plan, FmhaBf16Spec, MemoryPool, PinnedBuffer, Stream, bf16};
use runtime::kv::{KvCacheDType, KvStorageSpec};

use crate::{CudaBackend, Error, Result};

pub(super) fn prepare(
    backend: &CudaBackend,
    storage: KvStorageSpec,
    query_heads: usize,
    max_blocks: usize,
) -> Result<(bool, Option<FmhaBf16Plan>, Option<PagedFmhaPrefill>)> {
    let supported = matches!(storage.cache.dtype, KvCacheDType::Auto | KvCacheDType::BFloat16)
        && matches!(storage.key_head_dim, 64 | 128 | 256)
        && storage.value_head_dim == storage.key_head_dim;
    let plan = || {
        FmhaBf16Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            FmhaBf16Spec::new(
                query_heads,
                storage.kv_heads,
                storage.key_head_dim,
                storage.value_head_dim,
            )?,
        )
    };
    let fixed = (supported && storage.key_head_dim != 256).then(plan).transpose()?;
    let paged = (supported && storage.key_head_dim == 256)
        .then(|| {
            PagedFmhaPrefill::new(
                backend,
                plan()?,
                query_heads,
                max_blocks,
                storage.cache.block_size,
            )
        })
        .transpose()?;
    Ok((supported, fixed, paged))
}

#[derive(Debug)]
pub(super) struct PagedFmhaPrefill {
    plan: FmhaBf16Plan,
    query_starts: Metadata,
    token_counts: Metadata,
    context_starts: Metadata,
    softmax_lse: DeviceBuffer<f32>,
    stream: Stream,
    pool: MemoryPool,
    query_heads: usize,
    max_blocks: usize,
    block_size: usize,
}

impl PagedFmhaPrefill {
    pub(super) fn new(
        backend: &CudaBackend,
        plan: FmhaBf16Plan,
        query_heads: usize,
        max_blocks: usize,
        block_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            plan,
            query_starts: Metadata::new(backend, 2)?,
            token_counts: Metadata::new(backend, 1)?,
            context_starts: Metadata::new(backend, 2)?,
            softmax_lse: backend.inner.pool.allocate(&backend.inner.stream, query_heads)?,
            stream: backend.inner.stream.clone(),
            pool: backend.inner.pool.clone(),
            query_heads,
            max_blocks,
            block_size,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        block_table: &DeviceBuffer<u32>,
        query_tokens: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<()> {
        self.ensure_lse(query_tokens)?;
        self.query_starts.upload(&self.stream, &[0, u32::try_from(query_tokens)?])?;
        self.token_counts.upload(&self.stream, &[u32::try_from(context_tokens)?])?;
        self.context_starts.upload(&self.stream, &[0, u32::try_from(context_tokens)?])?;
        Ok(self.plan.execute_paged_varlen(
            &self.stream,
            query,
            key_pages,
            value_pages,
            output,
            &self.query_starts.device,
            &self.token_counts.device,
            &self.context_starts.device,
            block_table,
            &mut self.softmax_lse,
            1,
            query_tokens,
            query_tokens,
            context_tokens,
            self.max_blocks,
            self.block_size,
            scale,
        )?)
    }

    fn ensure_lse(&mut self, query_tokens: usize) -> Result<()> {
        let required = query_tokens
            .checked_mul(self.query_heads)
            .ok_or(Error::InvalidPagedKv("paged FMHA LSE workspace overflow"))?;
        if self.softmax_lse.len() < required {
            self.softmax_lse = self.pool.allocate(&self.stream, required)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Metadata {
    staging: PinnedBuffer<u32>,
    device: DeviceBuffer<u32>,
}

impl Metadata {
    fn new(backend: &CudaBackend, len: usize) -> Result<Self> {
        Ok(Self {
            staging: backend.inner.context.allocate_pinned(len)?,
            device: backend.inner.pool.allocate(&backend.inner.stream, len)?,
        })
    }

    fn upload(&mut self, stream: &Stream, values: &[u32]) -> Result<()> {
        self.staging.copy_from_slice(values)?;
        Ok(stream.copy_to_device(&mut self.staging, &mut self.device)?)
    }
}
