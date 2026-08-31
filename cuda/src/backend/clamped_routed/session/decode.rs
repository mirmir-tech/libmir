use mircuda::{DeviceBuffer, Graph, Stream, bf16};
use runtime::kv::BlockTable;
use uuid::Uuid;

use super::{ClampedRoutedSessionState, CudaClampedRoutedModelSession};
use crate::{
    Error, PagedKvCache, PagedPrefillBatch, Result,
    backend::clamped_routed::{
        CudaClampedRoutedModelTemplate,
        plan::{ClampedRoutedDecodeSignature, ClampedRoutedExecutionPlan},
        projection::ClampedRoutedEmbedding,
    },
};

pub(super) struct ClampedRoutedDecodeBatch {
    state: Option<DecodeState>,
    stream: Stream,
}

struct DecodeResources {
    template: CudaClampedRoutedModelTemplate,
    embedding: ClampedRoutedEmbedding,
    session: ClampedRoutedSessionState,
    plan: ClampedRoutedExecutionPlan,
    batch: PagedPrefillBatch,
    counts: Vec<usize>,
}

enum DecodeState {
    Direct(DecodeResources),
    Captured {
        graph: Graph<DecodeResources>,
        signature: ClampedRoutedDecodeSignature,
    },
}

impl ClampedRoutedDecodeBatch {
    pub(super) fn new(
        template: &CudaClampedRoutedModelTemplate,
        caches: &[PagedKvCache],
        rows: usize,
    ) -> Result<Self> {
        if rows == 0 {
            return Err(Error::InvalidDecoderKernel("clamped-routed decode batch is empty"));
        }
        let storage = template.config.storage(template.cache);
        let resources = DecodeResources {
            template: template.clone(),
            embedding: ClampedRoutedEmbedding::new(
                &template.backend,
                template.config,
                &template.embedding,
            )?,
            session: ClampedRoutedSessionState { caches: caches.to_vec() },
            plan: ClampedRoutedExecutionPlan::new(template, rows, crate::ExecutionPhase::Decode)?,
            batch: template.backend.prepare_paged_prefill_batch(
                storage,
                template.max_sequence_blocks,
                rows,
                rows,
            )?,
            counts: vec![1; rows],
        };
        Ok(Self {
            state: Some(DecodeState::Direct(resources)),
            stream: template.backend.inner.stream.clone(),
        })
    }

    pub(super) fn decode(
        &mut self,
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
        ring_slots: &[usize],
    ) -> Result<()> {
        self.with_resources_mut(|resources| resources.prepare(tokens, tables, starts, ring_slots))?;
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("clamped-routed decode state is unavailable"))?;
        self.state = Some(match state {
            DecodeState::Direct(resources) => self.capture_after_execute(resources)?,
            DecodeState::Captured { mut graph, signature } => {
                let next = graph.resources().signature();
                if next == signature {
                    graph.launch(&self.stream)?;
                    DecodeState::Captured { graph, signature }
                } else {
                    self.capture_after_execute(graph.into_resources())?
                }
            },
        });
        #[cfg(feature = "diagnostics")]
        self.publish_fingerprints()?;
        Ok(())
    }

    pub(super) fn hidden(&self) -> Result<&DeviceBuffer<bf16>> {
        Ok(
            match self
                .state
                .as_ref()
                .ok_or(Error::InvalidDecoderKernel("clamped-routed decode state is unavailable"))?
            {
                DecodeState::Direct(resources) => resources.plan.hidden(),
                DecodeState::Captured { graph, .. } => graph.resources().plan.hidden(),
            },
        )
    }

    fn capture_after_execute(&self, mut resources: DecodeResources) -> Result<DecodeState> {
        execute(&mut resources)?;
        let signature = resources.signature();
        let rows = resources.counts.len();
        let graph = self.stream.capture(resources, execute)?;
        tracing::debug!(rows, ?signature, "captured clamped-routed CUDA model decode batch");
        Ok(DecodeState::Captured { graph, signature })
    }

    fn with_resources_mut<T>(
        &mut self,
        operation: impl FnOnce(&mut DecodeResources) -> Result<T>,
    ) -> Result<T> {
        match self
            .state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("clamped-routed decode state is unavailable"))?
        {
            DecodeState::Direct(resources) => operation(resources),
            DecodeState::Captured { graph, .. } => graph.with_resources_mut(operation),
        }
    }

    #[cfg(feature = "diagnostics")]
    fn publish_fingerprints(&mut self) -> Result<()> {
        self.with_resources_mut(|resources| resources.plan.publish_fingerprints())
    }
}

impl DecodeResources {
    fn prepare(
        &mut self,
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
        ring_slots: &[usize],
    ) -> Result<()> {
        if tokens.len() != self.counts.len()
            || tables.len() != self.counts.len()
            || starts.len() != self.counts.len()
            || ring_slots.len() != self.counts.len()
        {
            return Err(Error::InvalidPagedKv("invalid clamped-routed decode geometry"));
        }
        self.batch.prepare_decode(tables, starts, &self.counts)?;
        if let Some(window) = self.template.max_sliding_window() {
            let ring_blocks = self
                .template
                .ring_blocks()
                .ok_or(Error::InvalidPagedKv("missing windowed KV ring geometry"))?;
            self.batch
                .prepare_ring(tables, starts, &self.counts, ring_slots, ring_blocks, window)?;
        }
        self.plan.upload_packed(&self.template, tokens)
    }

    fn signature(&self) -> ClampedRoutedDecodeSignature {
        self.plan.decode_signature(&self.batch)
    }
}

fn execute(resources: &mut DecodeResources) -> Result<()> {
    let DecodeResources {
        template,
        embedding,
        session,
        plan,
        batch,
        counts: _,
    } = resources;
    plan.execute_batch(template, session, embedding, batch)?;
    Ok(())
}

impl CudaClampedRoutedModelSession {
    pub(crate) fn decode_packed_chunk(
        &mut self,
        sessions: &[Uuid],
        tokens: &[u32],
        tables: &[&BlockTable],
    ) -> Result<()> {
        let starts = sessions
            .iter()
            .map(|session| self.positions.get(session).copied().unwrap_or_default())
            .collect::<Vec<_>>();
        let counts = vec![1; sessions.len()];
        self.validate_packed(sessions, tokens, tables, &starts, &counts)?;
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        let ring_slots = self.rings.acquire_many(sessions)?;
        let rows = sessions.len();
        if !self.decode_batches.contains_key(&rows) {
            self.decode_batches.insert(
                rows,
                ClampedRoutedDecodeBatch::new(&self.template, &self.state.caches, rows)?,
            );
        }
        self.decode_batches
            .get_mut(&rows)
            .ok_or(Error::InvalidDecoderKernel("missing clamped-routed decode batch"))?
            .decode(tokens, tables, &starts, &ring_slots)?;
        for (session, start) in sessions.iter().zip(starts) {
            self.positions.insert(*session, start + 1);
        }
        self.last_packed_decode = Some(rows);
        Ok(())
    }
}
