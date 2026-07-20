use mircuda::{DeviceBuffer, PinnedBuffer, bf16};

use super::CudaMoeModelTemplate;
use crate::{
    CudaDecodeBatch, CudaModelSessionConfig, CudaMoeModelSession, Error, Result,
    backend::model::{
        batch::DecodeBatchSource,
        layer::{DecoderLayerTemplate, SessionLayer},
        prefill::PrefillTokenBuffer,
    },
};

impl CudaMoeModelTemplate {
    /// Creates private activation, plan, K/V, and graph state for one session.
    pub fn instantiate(&self) -> Result<CudaMoeModelSession> {
        self.instantiate_with_config(CudaModelSessionConfig::default())
    }

    /// Creates one preallocated decode bucket without copying model weights.
    pub fn instantiate_decode_batch(&self, batch_size: usize) -> Result<CudaDecodeBatch> {
        let caches = self.allocate_shared_kv()?;
        self.instantiate_decode_batch_with_caches(batch_size, &caches)
    }

    pub(crate) fn instantiate_decode_batch_with_caches(
        &self,
        batch_size: usize,
        caches: &[crate::PagedKvCache],
    ) -> Result<CudaDecodeBatch> {
        let attention = self
            .layers
            .first()
            .ok_or(Error::InvalidDecoderKernel("CUDA model template requires layers"))?
            .attention();
        CudaDecodeBatch::new(
            &self.backend,
            DecodeBatchSource {
                embedding: self.embedding.clone(),
                final_norm: self.final_norm.clone(),
                output_head: &self.output_head,
                layers: &self.layers,
                caches,
                attention,
                vocab: self.decoder.vocab_size,
                hidden: self.decoder.hidden_size,
                embedding_scale: self.embedding_scale,
            },
            batch_size,
        )
    }

    /// Creates one session with an explicit prefill allocation policy.
    pub fn instantiate_with_config(
        &self,
        config: CudaModelSessionConfig,
    ) -> Result<CudaMoeModelSession> {
        let caches = self.allocate_shared_kv()?;
        self.instantiate_with_config_and_caches(config, &caches)
    }

    pub(crate) fn instantiate_with_config_and_caches(
        &self,
        config: CudaModelSessionConfig,
        caches: &[crate::PagedKvCache],
    ) -> Result<CudaMoeModelSession> {
        let config = config.validate()?;
        let hidden = self.decoder.hidden_size;
        let vocab = self.decoder.vocab_size;
        let first = self.backend.inner.pool.allocate::<bf16>(&self.backend.inner.stream, hidden)?;
        let second =
            self.backend.inner.pool.allocate::<bf16>(&self.backend.inner.stream, hidden)?;
        let layers = instantiate_layers(&self.layers, caches, &first, &second)?;
        let prefill_elements = config
            .prefill_chunk_tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("CUDA prefill activation size overflow"))?;
        let prefill_first = self
            .backend
            .inner
            .pool
            .allocate::<bf16>(&self.backend.inner.stream, prefill_elements)?;
        let prefill_second = self
            .backend
            .inner
            .pool
            .allocate::<bf16>(&self.backend.inner.stream, prefill_elements)?;
        let logits = self.backend.inner.pool.allocate::<bf16>(&self.backend.inner.stream, vocab)?;
        let input_token = self.backend.inner.pool.allocate::<u32>(&self.backend.inner.stream, 1)?;
        let token_staging: PinnedBuffer<u32> = self.backend.inner.context.allocate_pinned(1)?;
        let epsilon = self
            .layers
            .first()
            .ok_or(Error::InvalidDecoderKernel("CUDA model template requires layers"))?
            .attention()
            .rms_norm_epsilon;
        CudaMoeModelSession::new(
            self.backend.clone(),
            hidden,
            self.backend.prepare_bf16_embedding(vocab, hidden, self.embedding_scale)?,
            self.backend.prepare_rms_norm_bf16(1, hidden, epsilon)?,
            self.output_head.instantiate(&self.backend)?,
            self.embedding.clone(),
            self.final_norm.clone(),
            layers,
            self.backend.prepare_device_sampler_bf16(vocab)?,
            self.backend.inner.stream.clone(),
            input_token,
            token_staging,
            PrefillTokenBuffer::new(&self.backend, config)?,
            crate::kernels::SelectRowBf16::compile(&self.backend.inner.compiler, hidden)?,
            first,
            second,
            prefill_first,
            prefill_second,
            logits,
        )
    }
}

fn instantiate_layers(
    templates: &[DecoderLayerTemplate],
    caches: &[crate::PagedKvCache],
    first: &DeviceBuffer<bf16>,
    second: &DeviceBuffer<bf16>,
) -> Result<Vec<SessionLayer>> {
    if templates.len() != caches.len() {
        return Err(Error::InvalidDecoderKernel("CUDA session cache count differs from layers"));
    }
    templates
        .iter()
        .zip(caches)
        .enumerate()
        .map(|(index, (template, cache))| {
            let (input, output) = if index.is_multiple_of(2) {
                (first, second)
            } else {
                (second, first)
            };
            template.instantiate(input, output, cache.clone())
        })
        .collect()
}
