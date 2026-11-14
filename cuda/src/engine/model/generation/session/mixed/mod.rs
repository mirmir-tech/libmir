use std::collections::HashMap;

use uuid::Uuid;

use super::super::super::{GenerationExecution, SpatialVisionPrefill};
use crate::{
    CudaBackend, CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, Error, Result,
    backend::{CudaSharedRoutedDecodeBatch, CudaSharedRoutedPrefillBatch},
    engine::execution::{Output, generation_output},
};

mod batch;
mod checkpoint;
mod combined;
mod drafting;
mod memory;
mod prefill_budget;
mod speculation;
mod vision;
mod warm;
use checkpoint::PrefixCheckpoints;
use prefill_budget::reusable_prefill_budget;

pub(in crate::engine) struct MixedMixerExecution {
    template: CudaSharedRoutedModelTemplate,
    caches: Vec<Option<crate::PagedKvCache>>,
    prefill_chunk_tokens: usize,
    sessions: HashMap<Uuid, CudaSharedRoutedModelSession>,
    decode_batches: HashMap<usize, CudaSharedRoutedDecodeBatch>,
    prefill_batches: HashMap<(usize, usize), CudaSharedRoutedPrefillBatch>,
    checkpoints: PrefixCheckpoints,
    combined: combined::CombinedBatches,
    /// Pool bytes one live session cost during concurrency warm-up.
    session_bytes: Option<u64>,
    drafting: Option<speculation::Drafting>,
}

impl MixedMixerExecution {
    pub(in crate::engine) fn new(
        template: CudaSharedRoutedModelTemplate,
        prefill_chunk_tokens: usize,
        speculation: Option<runtime::speculation::Speculation>,
    ) -> Result<Self> {
        if prefill_chunk_tokens == 0 {
            return Err(Error::InvalidDecoderKernel("shared-routed prefill chunk is empty"));
        }
        let caches = template.allocate_shared_kv()?;
        let checkpoints = memory::prefix_checkpoints(&caches);
        Ok(Self {
            template,
            caches,
            prefill_chunk_tokens,
            sessions: HashMap::new(),
            decode_batches: HashMap::new(),
            prefill_batches: HashMap::new(),
            checkpoints,
            combined: combined::CombinedBatches::default(),
            session_bytes: None,
            drafting: speculation.map(speculation::Drafting::new),
        })
    }
}

impl GenerationExecution for MixedMixerExecution {
    fn prefill_schedule(&self) -> crate::CudaPrefillSchedule {
        // Routed runners qualify once their rows can join combined ragged
        // steps.
        if self.template.decoder().num_experts.is_none()
            || self.template.supports_combined_generation()
        {
            crate::CudaPrefillSchedule::CompletionFirst
        } else {
            crate::CudaPrefillSchedule::RoundRobin
        }
    }

    fn prefix_replay_tokens(&self) -> Option<usize> {
        Some(usize::MAX)
    }

    fn prefix_admission(&self) -> Option<(usize, usize, usize)> {
        Some((usize::MAX, 0, 0))
    }

    fn restore_prefix(
        &mut self,
        request: &runtime::backend::PrefillRequest,
        minimum: usize,
        maximum: usize,
    ) -> Result<Option<usize>> {
        self.restore_prefix_session(request, minimum, maximum)
    }

    fn terminal_cache_checkpoint(
        &self,
        request: &runtime::backend::PrefillRequest,
    ) -> Option<usize> {
        request.terminal_cache_checkpoint().filter(|tokens| {
            self.prefill_schedule() != crate::CudaPrefillSchedule::CompletionFirst || *tokens >= 128
        })
    }

    fn splits_at_terminal_checkpoint(&self) -> bool {
        false
    }

    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining.min(self.prefill_chunk_tokens)
    }

    fn interleaved_prefill_budget(&self, budget: usize) -> usize {
        reusable_prefill_budget(budget, self.prefill_chunk_tokens)
    }

    fn prefill_chunk(
        &mut self,
        backend: &CudaBackend,
        request: &runtime::backend::PrefillRequest,
        tokens: &[u32],
        offset: usize,
        table: &runtime::kv::BlockTable,
        final_chunk: bool,
    ) -> Result<Option<Output>> {
        if offset == 0 {
            let session = self.template.instantiate_with_caches(&self.caches)?;
            self.sessions.insert(request.session_id, session);
        }
        self.arm_terminal_checkpoint(request, offset, tokens.len())?;
        if self.drafting.is_some() {
            return self
                .prefill_drafting_chunk(backend, request, tokens, offset, table, final_chunk);
        }
        let session = required(&mut self.sessions, request.session_id)?;
        session.prefill(request.session_id, tokens, table)?;
        self.checkpoint_prefix(request)?;
        if final_chunk {
            let session = required(&mut self.sessions, request.session_id)?;
            generation_output(backend, session, request.sampling_logits.clone()).map(Some)
        } else {
            Ok(None)
        }
    }

    fn prefill_batch_chunk(
        &mut self,
        backend: &CudaBackend,
        chunks: &[super::super::PrefillChunk<'_>],
    ) -> Result<Vec<Option<Output>>> {
        if self.drafting.is_some() {
            return self.prefill_drafting_chunks(backend, chunks);
        }
        batch::prefill(self, backend, chunks)
    }

    fn decode(
        &mut self,
        backend: &CudaBackend,
        request: &runtime::backend::DecodeRequest,
        _use_device_token: bool,
    ) -> Result<Output> {
        let session = required(&mut self.sessions, request.session_id)?;
        if request.speculation.is_some() && self.drafting.is_some() && session.predicts() {
            return self.decode_speculative(request);
        }
        session.decode(request.session_id, request.token_id, &request.block_table)?;
        generation_output(backend, session, request.sampling_logits.clone())
    }

    fn decode_batch(
        &mut self,
        backend: &CudaBackend,
        sequences: &[runtime::backend::DecodeSequence],
    ) -> Result<Option<Vec<Output>>> {
        if self.drafting.is_some() {
            if sequences.len() < 2 {
                return Ok(None);
            }
            if let Some(outputs) = drafting::execute(self, &[], sequences)? {
                return Ok(Some(outputs.decode));
            }
            self.forget_drafts(sequences);
        }
        batch::decode(self, backend, sequences)
    }

    fn prefill_decode_batch(
        &mut self,
        backend: &CudaBackend,
        chunks: &[super::super::super::PrefillChunk<'_>],
        decode: &[runtime::backend::DecodeSequence],
    ) -> Result<Option<crate::engine::model::generation::CombinedOutputs>> {
        if self.drafting.is_some() {
            return drafting::execute(self, chunks, decode);
        }
        combined::execute(self, backend, chunks, decode)
    }

    fn clear_sessions(&mut self) {
        self.combined.clear();
        self.sessions.clear();
        self.decode_batches.clear();
        self.prefill_batches.clear();
        self.checkpoints.clear();
    }

    fn retention_headroom_bytes(&self) -> Result<u64> {
        memory::retention_headroom_bytes(self)
    }

    fn warm_concurrency(&mut self, backend: &CudaBackend, rows: usize) -> Result<()> {
        warm::warm_concurrency(self, backend, rows)
    }

    fn session_bytes(&self) -> Option<u64> {
        self.session_bytes
    }

    fn retained_shape_bytes(&self) -> Result<u64> {
        // Freed shapes are kept below the threshold on purpose: a lower anchor
        // released them and left less host memory, not more (2026-09-22).
        Ok(0)
    }

    fn resize_kv_cache(&mut self, cache: runtime::kv::CacheConfig) -> Result<bool> {
        memory::resize_kv_cache(self, cache)
    }

    fn release_session(&mut self, session: Uuid) {
        self.sessions.remove(&session);
    }

    fn prefill_spatial_vision(
        &mut self,
        backend: &CudaBackend,
        input: SpatialVisionPrefill<'_>,
    ) -> Result<Output> {
        self.prefill_vision_session(backend, input)
    }
}

fn required(
    sessions: &mut HashMap<Uuid, CudaSharedRoutedModelSession>,
    id: Uuid,
) -> Result<&mut CudaSharedRoutedModelSession> {
    sessions
        .get_mut(&id)
        .ok_or_else(|| Error::State("CUDA decoder session is not initialized".into()))
}
