use mircuda::{DeviceBuffer, Graph, PinnedBuffer, Stream, bf16};
use runtime::{backend::SamplingLogits, kv::BlockTable};

use super::layer::{BatchedLayer, DecoderLayerTemplate};
use crate::{
    Bf16Embedding, CudaBackend, CudaTensor, DecodeAttentionConfig, DeviceBatchSamplerBf16, Error,
    PagedDecodeBatch, PagedKvCache, Result, RmsNormBf16,
    backend::output::{CudaBatchOutputHead, CudaOutputHeadTemplate},
};

/// Fixed-capacity model decode bucket with device-resident row state.
pub struct CudaDecodeBatch {
    state: Option<BatchState>,
    stream: Stream,
    batch_size: usize,
}

struct BatchResources {
    embedding: Bf16Embedding,
    embedding_weight: CudaTensor,
    tokens: DeviceBuffer<u32>,
    token_staging: PinnedBuffer<u32>,
    paging: PagedDecodeBatch,
    layers: Vec<BatchedLayer>,
    final_norm: RmsNormBf16,
    final_norm_weight: CudaTensor,
    output_head: CudaBatchOutputHead,
    sampler: DeviceBatchSamplerBf16,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
}

enum BatchState {
    Direct(BatchResources),
    Captured(Graph<BatchResources>),
}

pub(super) struct DecodeBatchSource<'a> {
    pub(super) embedding: CudaTensor,
    pub(super) final_norm: CudaTensor,
    pub(super) output_head: &'a CudaOutputHeadTemplate,
    pub(super) layers: &'a [DecoderLayerTemplate],
    pub(super) caches: &'a [PagedKvCache],
    pub(super) attention: DecodeAttentionConfig,
    pub(super) vocab: usize,
    pub(super) hidden: usize,
    pub(super) embedding_scale: f32,
}

impl CudaDecodeBatch {
    pub(super) fn new(
        backend: &CudaBackend,
        source: DecodeBatchSource<'_>,
        batch_size: usize,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidDecoderKernel("CUDA decode batch is empty"));
        }
        let DecodeBatchSource {
            embedding,
            final_norm,
            output_head,
            layers,
            caches,
            attention,
            vocab,
            hidden,
            embedding_scale,
        } = source;
        if layers.len() != caches.len() {
            return Err(Error::InvalidDecoderKernel("CUDA batch cache count differs from layers"));
        }
        let elements = batch_size
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("CUDA decode batch activation overflow"))?;
        let resources = BatchResources {
            embedding: backend.prepare_bf16_embedding(vocab, hidden, embedding_scale)?,
            embedding_weight: embedding,
            tokens: backend.inner.pool.allocate(&backend.inner.stream, batch_size)?,
            token_staging: backend.inner.context.allocate_pinned(batch_size)?,
            paging: backend.prepare_paged_decode_batch(
                attention.cache,
                attention.max_sequence_blocks,
                batch_size,
            )?,
            layers: layers
                .iter()
                .zip(caches)
                .map(|(layer, cache)| layer.instantiate_batch(batch_size, cache.clone()))
                .collect::<Result<Vec<_>>>()?,
            final_norm: backend.prepare_rms_norm_bf16(
                batch_size,
                hidden,
                attention.rms_norm_epsilon,
            )?,
            final_norm_weight: final_norm,
            output_head: CudaBatchOutputHead::new(backend, output_head, batch_size)?,
            sampler: backend.prepare_device_batch_sampler_bf16(vocab, batch_size)?,
            first: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            second: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            normalized: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            logits: backend.inner.pool.allocate(
                &backend.inner.stream,
                batch_size
                    .checked_mul(vocab)
                    .ok_or(Error::InvalidDecoderKernel("CUDA decode batch logits overflow"))?,
            )?,
        };
        Ok(Self {
            state: Some(BatchState::Direct(resources)),
            stream: backend.inner.stream.clone(),
            batch_size,
        })
    }

    /// Enqueues a complete model decode for one fixed bucket.
    pub fn decode(
        &mut self,
        tokens: &[u32],
        tables: &[&BlockTable],
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prepare(tokens, tables)?;
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("CUDA batch state is unavailable"))?;
        self.state = Some(match state {
            BatchState::Direct(mut resources) => {
                execute(&mut resources)?;
                let graph = self.stream.capture(resources, execute)?;
                tracing::debug!(rows = self.batch_size, "captured CUDA model decode batch");
                BatchState::Captured(graph)
            },
            BatchState::Captured(mut graph) => {
                graph.launch(&self.stream)?;
                BatchState::Captured(graph)
            },
        });
        self.logits_result()
    }

    /// Samples each logits row on the device using its own bounded policy.
    pub fn sample(&mut self, policies: &[SamplingLogits]) -> Result<&DeviceBuffer<u32>> {
        self.with_state_mut(|resources| {
            resources.sampler.sample(&resources.logits, policies)?;
            Ok(())
        })?;
        Ok(self.state()?.sampler.selected())
    }

    /// Uploads one exact bucket and produces its embedded input activations.
    pub fn prepare(&mut self, tokens: &[u32], tables: &[&BlockTable]) -> Result<()> {
        if tokens.len() != self.batch_size || tables.len() != self.batch_size {
            return Err(Error::InvalidDecoderKernel("CUDA decode rows differ from bucket size"));
        }
        let rows = self.batch_size;
        let stream = self.stream.clone();
        self.with_state_mut(|resources| {
            for token in tokens {
                resources.embedding.validate_token(*token)?;
            }
            resources.token_staging.copy_from_slice(tokens)?;
            stream.copy_to_device(&mut resources.token_staging, &mut resources.tokens)?;
            resources.paging.prepare(tables)?;
            resources.embedding.execute_batch(
                &resources.tokens,
                0,
                rows,
                &resources.embedding_weight,
                &mut resources.first,
            )
        })
    }

    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn logits(&self) -> Result<&DeviceBuffer<bf16>> {
        self.logits_result()
    }

    fn logits_result(&self) -> Result<&DeviceBuffer<bf16>> {
        Ok(&self.state()?.logits)
    }

    fn state(&self) -> Result<&BatchResources> {
        match self
            .state
            .as_ref()
            .ok_or(Error::InvalidDecoderKernel("CUDA batch state is unavailable"))?
        {
            BatchState::Direct(resources) => Ok(resources),
            BatchState::Captured(graph) => Ok(graph.resources()),
        }
    }

    fn with_state_mut<T>(
        &mut self,
        operation: impl FnOnce(&mut BatchResources) -> Result<T>,
    ) -> Result<T> {
        match self
            .state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("CUDA batch state is unavailable"))?
        {
            BatchState::Direct(resources) => operation(resources),
            BatchState::Captured(graph) => graph.with_resources_mut(operation),
        }
    }
}

fn execute(resources: &mut BatchResources) -> Result<()> {
    for (index, layer) in resources.layers.iter_mut().enumerate() {
        if index.is_multiple_of(2) {
            layer.execute(&resources.first, &resources.paging, &mut resources.second)?;
        } else {
            layer.execute(&resources.second, &resources.paging, &mut resources.first)?;
        }
    }
    let source = if resources.layers.len().is_multiple_of(2) {
        &resources.first
    } else {
        &resources.second
    };
    resources.final_norm.execute(
        source,
        &resources.final_norm_weight,
        &mut resources.normalized,
    )?;
    resources.output_head.execute(&resources.normalized, &mut resources.logits)
}
