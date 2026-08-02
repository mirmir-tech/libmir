use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::{
    CudaClampedRoutedModelTemplate, layer::ClampedRoutedLayerExecution,
    scratch::ClampedRoutedScratch, session::ClampedRoutedSessionState,
    weights::ClampedRoutedExpertWeights,
};
use crate::{
    Error, ExecutionPhase, PagedPrefillBatch, Result,
    backend::{clamped_routed::projection::ClampedRoutedEmbedding, linear::SelectedDenseMoeBf16},
    kernels::{ClampedRoutedBatchSplitDecode, ClampedRoutedSplitDecode, DenseGatedActivation},
};

mod decode;
mod upload;

pub(super) use decode::ClampedRoutedDecodeSignature;

pub(super) struct ClampedRoutedExecutionPlan {
    tokens: usize,
    token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    position_staging: PinnedBuffer<u32>,
    positions: DeviceBuffer<u32>,
    table_staging: PinnedBuffer<u32>,
    table_device: DeviceBuffer<u32>,
    table_snapshot: Vec<u32>,
    ring_table_staging: PinnedBuffer<u32>,
    ring_table_device: DeviceBuffer<u32>,
    ring_table_snapshot: Vec<u32>,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    layers: Vec<ClampedRoutedLayerExecution>,
    scratch: ClampedRoutedScratch,
    dense_experts: Option<SelectedDenseMoeBf16>,
    split_decode: Option<ClampedRoutedSplitDecode>,
    batch_split_decode: Option<ClampedRoutedBatchSplitDecode>,
}

impl ClampedRoutedExecutionPlan {
    pub(super) fn new(
        template: &CudaClampedRoutedModelTemplate,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("empty clamped-routed execution plan"));
        }
        let backend = &template.backend;
        let config = template.config;
        let storage = config.storage(template.cache);
        let elements = tokens
            .checked_mul(config.hidden)
            .ok_or(Error::InvalidDecoderKernel("clamped-routed activation size overflow"))?;
        let layers = template
            .layers
            .iter()
            .map(|layer| layer.prepare(tokens, storage, phase))
            .collect::<Result<Vec<_>>>()?;
        let dense_experts = template
            .layers
            .iter()
            .find_map(|layer| match &layer.weights().experts {
                ClampedRoutedExpertWeights::Dense(weights) => Some(weights.as_ref()),
                ClampedRoutedExpertWeights::Native(_) | ClampedRoutedExpertWeights::Mlx(_) => None,
            })
            .map(|weights| {
                SelectedDenseMoeBf16::new(
                    backend,
                    tokens,
                    config.top_k,
                    weights,
                    DenseGatedActivation::clamped_silu(1.702, config.swiglu_limit, 1.0),
                )
            })
            .transpose()?;
        let split_decode = if tokens == 1 {
            ClampedRoutedSplitDecode::compile(
                backend,
                storage,
                config.query_heads,
                template.max_sequence_blocks,
            )?
        } else {
            None
        };
        Ok(Self {
            tokens,
            token_staging: backend.inner.context.allocate_pinned(tokens)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            position_staging: backend.inner.context.allocate_pinned(tokens)?,
            positions: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            table_staging: backend.inner.context.allocate_pinned(template.max_sequence_blocks)?,
            table_device: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, template.max_sequence_blocks)?,
            table_snapshot: vec![u32::MAX; template.max_sequence_blocks],
            ring_table_staging: backend
                .inner
                .context
                .allocate_pinned(template.max_sequence_blocks)?,
            ring_table_device: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, template.max_sequence_blocks)?,
            ring_table_snapshot: vec![u32::MAX; template.max_sequence_blocks],
            first: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            second: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            layers,
            scratch: ClampedRoutedScratch::new(
                backend,
                config,
                tokens,
                template.layers.iter().any(|layer| {
                    matches!(
                        &layer.weights().experts,
                        ClampedRoutedExpertWeights::Native(_) | ClampedRoutedExpertWeights::Mlx(_)
                    )
                }),
            )?,
            dense_experts,
            split_decode,
            batch_split_decode: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute<'a>(
        &'a mut self,
        template: &CudaClampedRoutedModelTemplate,
        embedding: &ClampedRoutedEmbedding,
        state: &mut ClampedRoutedSessionState,
        session: Uuid,
        ring_slot: usize,
        table: &BlockTable,
        start: usize,
        cached_until: usize,
    ) -> Result<&'a DeviceBuffer<bf16>> {
        embedding.execute_batch(
            &self.token_ids,
            self.tokens,
            &template.embedding,
            &mut self.first,
        )?;
        if self.layers.len() != state.caches.len() {
            return Err(Error::InvalidPagedKv("clamped-routed cache layer count mismatch"));
        }
        let first = self
            .layers
            .first()
            .ok_or(Error::InvalidExecutionPlan("decoder has no layers"))?;
        first.prepare_rope(&self.positions, &mut self.scratch)?;
        let (layers, scratch, dense_experts, split_decode) = (
            &mut self.layers,
            &mut self.scratch,
            &mut self.dense_experts,
            &mut self.split_decode,
        );
        for (index, (layer, cache)) in layers.iter_mut().zip(&mut state.caches).enumerate() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            let write = KvWritePlan::prefill(session, index, table, start, self.tokens)?;
            layer.execute(
                &template.layers[index],
                input,
                cache,
                &write,
                table,
                &self.table_device,
                &self.ring_table_device,
                ring_slot,
                start,
                cached_until,
                scratch,
                dense_experts,
                split_decode,
                output,
            )?;
        }
        Ok(self.hidden())
    }

    pub(super) fn execute_batch(
        &mut self,
        template: &CudaClampedRoutedModelTemplate,
        state: &mut ClampedRoutedSessionState,
        embedding: &ClampedRoutedEmbedding,
        batch: &PagedPrefillBatch,
    ) -> Result<&DeviceBuffer<bf16>> {
        embedding.execute_batch(
            &self.token_ids,
            self.tokens,
            &template.embedding,
            &mut self.first,
        )?;
        if self.layers.len() != state.caches.len() || batch.tokens() != self.tokens {
            return Err(Error::InvalidPagedKv(
                "packed clamped-routed cache or token geometry differs",
            ));
        }
        if batch.max_query_tokens() == 1 && self.batch_split_decode.is_none() {
            self.batch_split_decode = ClampedRoutedBatchSplitDecode::compile(
                &template.backend,
                template.config.storage(template.cache),
                template.config.query_heads,
                template.max_sequence_blocks,
                batch.active(),
            )?;
        }
        let first = self
            .layers
            .first()
            .ok_or(Error::InvalidExecutionPlan("decoder has no layers"))?;
        first.prepare_rope(batch.positions(), &mut self.scratch)?;
        let (layers, scratch, dense_experts, batch_split_decode) = (
            &mut self.layers,
            &mut self.scratch,
            &mut self.dense_experts,
            &mut self.batch_split_decode,
        );
        for (index, (layer, cache)) in layers.iter_mut().zip(&mut state.caches).enumerate() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            layer.execute_batch(
                &template.layers[index],
                input,
                cache,
                batch,
                scratch,
                dense_experts,
                batch_split_decode,
                output,
            )?;
        }
        Ok(self.hidden())
    }

    pub(super) fn hidden(&self) -> &DeviceBuffer<bf16> {
        if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        }
    }
}
