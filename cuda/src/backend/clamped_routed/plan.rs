use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::{
    CudaClampedRoutedModelTemplate, layer::ClampedRoutedLayerExecution,
    session::ClampedRoutedSessionState,
};
use crate::{Error, Result, backend::clamped_routed::projection::ClampedRoutedEmbedding};

pub(super) struct ClampedRoutedExecutionPlan {
    tokens: usize,
    token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    table_staging: PinnedBuffer<u32>,
    table_device: DeviceBuffer<u32>,
    table_snapshot: Vec<u32>,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    layers: Vec<ClampedRoutedLayerExecution>,
}

impl ClampedRoutedExecutionPlan {
    pub(super) fn new(template: &CudaClampedRoutedModelTemplate, tokens: usize) -> Result<Self> {
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
            .map(|layer| layer.prepare(tokens, storage))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tokens,
            token_staging: backend.inner.context.allocate_pinned(tokens)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            table_staging: backend.inner.context.allocate_pinned(template.max_sequence_blocks)?,
            table_device: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, template.max_sequence_blocks)?,
            table_snapshot: vec![u32::MAX; template.max_sequence_blocks],
            first: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            second: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            layers,
        })
    }

    pub(super) fn upload(
        &mut self,
        template: &CudaClampedRoutedModelTemplate,
        tokens: &[u32],
        table: &BlockTable,
    ) -> Result<()> {
        if tokens.len() != self.tokens || table.blocks().len() > self.table_snapshot.len() {
            return Err(Error::InvalidPagedKv(
                "clamped-routed plan input differs from its geometry",
            ));
        }
        self.token_staging.copy_from_slice(tokens)?;
        self.table_snapshot.fill(u32::MAX);
        for (target, block) in self.table_snapshot.iter_mut().zip(table.blocks()) {
            *target = block.0;
        }
        self.table_staging.copy_from_slice(&self.table_snapshot)?;
        let stream = &template.backend.inner.stream;
        stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        stream.copy_to_device(&mut self.table_staging, &mut self.table_device)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute<'a>(
        &'a mut self,
        template: &CudaClampedRoutedModelTemplate,
        embedding: &ClampedRoutedEmbedding,
        state: &mut ClampedRoutedSessionState,
        session: Uuid,
        table: &BlockTable,
        start: usize,
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
        for (index, (layer, cache)) in self.layers.iter_mut().zip(&mut state.caches).enumerate() {
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
                start,
                output,
            )?;
        }
        Ok(if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        })
    }
}
