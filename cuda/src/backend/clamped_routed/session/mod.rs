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

mod batch;
mod checkpoint;
mod decode;
mod output;
mod ring;
use checkpoint::PrefixCheckpoints;
use decode::ClampedRoutedDecodeBatch;
use ring::SessionRings;

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
    positions: HashMap<Uuid, usize>,
    rings: SessionRings,
    checkpoints: PrefixCheckpoints,
    decode_batches: HashMap<usize, ClampedRoutedDecodeBatch>,
    packed_batches: HashMap<(usize, usize), crate::PagedPrefillBatch>,
    packed_outputs: HashMap<usize, output::ClampedRoutedPackedOutput>,
    last_packed_decode: Option<usize>,
}

pub(super) struct ClampedRoutedSessionState {
    pub(super) caches: Vec<PagedKvCache>,
}

impl CudaClampedRoutedModelSession {
    pub(super) fn new(
        template: &CudaClampedRoutedModelTemplate,
        caches: &[PagedKvCache],
    ) -> Result<Self> {
        let backend = &template.backend;
        let config = template.config;
        let storage = config.storage(template.cache);
        if caches.len() != template.decoder.num_hidden_layers
            || caches.iter().enumerate().any(|(layer, cache)| {
                cache.layer() != layer
                    || cache.storage_spec() != storage
                    || cache.is_windowed() != template.layers[layer].window().is_some()
            })
        {
            return Err(Error::InvalidPagedKv(
                "clamped-routed shared cache differs from model geometry",
            ));
        }
        Ok(Self {
            template: template.clone(),
            embedding: ClampedRoutedEmbedding::new(backend, config, &template.embedding)?,
            state: ClampedRoutedSessionState { caches: caches.to_vec() },
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
            positions: HashMap::new(),
            rings: SessionRings::new(template.ring_sessions),
            checkpoints: PrefixCheckpoints::new(
                template.ring_sessions,
                template.checkpoint_slots()?,
            ),
            decode_batches: HashMap::new(),
            packed_batches: HashMap::new(),
            packed_outputs: HashMap::new(),
            last_packed_decode: None,
        })
    }

    pub fn prefill(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prefill_with_sampling(session, tokens, table, 0, SamplingLogits::Full)
    }

    pub(crate) fn prefill_with_sampling(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        table: &BlockTable,
        cached_until: usize,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        let position = self.positions.get(&session).copied().unwrap_or_default();
        let end = position
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
        let ring_slot = self.rings.acquire(session)?;
        if !self.plans.contains_key(&tokens.len()) {
            self.plans.clear();
            let plan = ClampedRoutedExecutionPlan::new(
                &self.template,
                tokens.len(),
                crate::ExecutionPhase::Prefill,
            )?;
            self.plans.insert(tokens.len(), plan);
        }
        let plan = self
            .plans
            .get_mut(&tokens.len())
            .ok_or(Error::InvalidDecoderKernel("missing clamped-routed execution plan"))?;
        plan.upload(&self.template, tokens, table, position, ring_slot)?;
        let hidden = plan.execute(
            &self.template, &self.embedding, &mut self.state, session, ring_slot, table, position,
            cached_until,
        )?;
        self.select.execute(
            &self.template.backend.inner.stream,
            hidden,
            &mut self.last_hidden,
            tokens.len() - 1,
            tokens.len(),
        )?;
        #[cfg(feature = "diagnostics")]
        plan.publish_fingerprints()?;
        self.final_norm.execute(
            &self.last_hidden,
            &self.template.final_norm,
            &mut self.normalized,
        )?;
        self.output.execute(&self.normalized, &mut self.logits, sampling)?;
        self.positions.insert(session, end);
        Ok(&self.logits)
    }

    #[must_use]
    pub fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining.min(crate::backend::model::DEFAULT_PREFILL_CHUNK_TOKENS)
    }

    pub fn decode(
        &mut self,
        session: Uuid,
        token: u32,
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.decode_with_sampling(session, token, table, SamplingLogits::Full)
    }

    pub(crate) fn decode_with_sampling(
        &mut self,
        session: Uuid,
        token: u32,
        table: &BlockTable,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.decode_packed_chunk(&[session], &[token], &[table])?;
        self.finish_packed_prefill_row(0, 1, sampling)?;
        Ok(&self.logits)
    }

    pub fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        self.sampler.sample(&self.logits, policy)
    }

    pub(crate) fn begin_chunk(&mut self, session: Uuid, offset: usize) -> Result<()> {
        if let Some(position) = self.positions.get(&session)
            && *position != offset
        {
            return Err(Error::InvalidPagedKv(
                "clamped-routed chunk offset differs from session state",
            ));
        }
        self.rings.acquire(session)?;
        self.positions.entry(session).or_insert(offset);
        Ok(())
    }

    #[must_use]
    pub const fn logits(&self) -> &DeviceBuffer<bf16> {
        &self.logits
    }

    pub(crate) fn clear_sessions(&mut self) {
        self.positions.clear();
        self.rings.clear();
        self.checkpoints.clear();
    }

    pub(crate) fn release_session(&mut self, session: Uuid) {
        self.positions.remove(&session);
        self.rings.release(session);
    }
}
