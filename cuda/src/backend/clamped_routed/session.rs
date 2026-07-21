use std::collections::HashMap;

use mircuda::{DeviceBuffer, bf16};
use runtime::{backend::SamplingLogits, kv::BlockTable};
use uuid::Uuid;

use super::{
    CudaClampedRoutedModelTemplate,
    plan::ClampedRoutedExecutionPlan,
    projection::{ClampedRoutedEmbedding, ClampedRoutedOutput},
};
use crate::{DeviceSamplerBf16, Error, PagedKvCache, Result, RmsNormBf16, kernels::SelectRowBf16};

pub struct CudaClampedRoutedModelSession {
    template: CudaClampedRoutedModelTemplate,
    embedding: ClampedRoutedEmbedding,
    state: ClampedRoutedSessionState,
    plans: HashMap<usize, ClampedRoutedExecutionPlan>,
    select: SelectRowBf16,
    final_norm: RmsNormBf16,
    output: ClampedRoutedOutput,
    sampler: DeviceSamplerBf16,
    last_hidden: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
    position: usize,
}

pub(super) struct ClampedRoutedSessionState {
    pub(super) caches: Vec<PagedKvCache>,
}

impl CudaClampedRoutedModelSession {
    const PREFILL_CHUNK_TOKENS: usize = 256;

    pub(super) fn new(template: &CudaClampedRoutedModelTemplate) -> Result<Self> {
        let backend = &template.backend;
        let config = template.config;
        let storage = config.storage(template.cache);
        let caches = (0..template.decoder.num_hidden_layers)
            .map(|layer| backend.prepare_paged_kv(layer, storage))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            template: template.clone(),
            embedding: ClampedRoutedEmbedding::new(backend, config, &template.embedding)?,
            state: ClampedRoutedSessionState { caches },
            plans: HashMap::new(),
            select: SelectRowBf16::compile(&backend.inner.compiler, config.hidden)?,
            final_norm: RmsNormBf16::new(backend, 1, config.hidden, config.epsilon)?,
            output: ClampedRoutedOutput::new(backend, config, &template.output)?,
            sampler: backend.prepare_device_sampler_bf16(template.decoder.vocab_size)?,
            last_hidden: backend.inner.pool.allocate(&backend.inner.stream, config.hidden)?,
            normalized: backend.inner.pool.allocate(&backend.inner.stream, config.hidden)?,
            logits: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, template.decoder.vocab_size)?,
            position: 0,
        })
    }

    pub fn prefill(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        let end = self
            .position
            .checked_add(tokens.len())
            .ok_or(Error::InvalidPagedKv("clamped-routed session position overflow"))?;
        if tokens.is_empty()
            || table.token_len() != end
            || table.block_size() != Some(self.template.cache.block_size)
        {
            return Err(Error::InvalidPagedKv("clamped-routed prompt differs from session state"));
        }
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        if !self.plans.contains_key(&tokens.len()) {
            self.plans.insert(
                tokens.len(),
                ClampedRoutedExecutionPlan::new(&self.template, tokens.len())?,
            );
        }
        let plan = self
            .plans
            .get_mut(&tokens.len())
            .ok_or(Error::InvalidDecoderKernel("missing clamped-routed execution plan"))?;
        plan.upload(&self.template, tokens, table)?;
        let hidden = plan.execute(
            &self.template, &self.embedding, &mut self.state, session, table, self.position,
        )?;
        self.select.execute(
            &self.template.backend.inner.stream,
            hidden,
            &mut self.last_hidden,
            tokens.len() - 1,
            tokens.len(),
        )?;
        self.final_norm.execute(
            &self.last_hidden,
            &self.template.final_norm,
            &mut self.normalized,
        )?;
        self.output.execute(&self.normalized, &self.template.output, &mut self.logits)?;
        self.position = end;
        Ok(&self.logits)
    }

    #[must_use]
    pub fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining.min(Self::PREFILL_CHUNK_TOKENS)
    }

    pub fn decode(
        &mut self,
        session: Uuid,
        token: u32,
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prefill(session, &[token], table)
    }

    pub fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        self.sampler.sample(&self.logits, policy)
    }

    #[must_use]
    pub const fn logits(&self) -> &DeviceBuffer<bf16> {
        &self.logits
    }
}
