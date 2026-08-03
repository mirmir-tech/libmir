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
use checkpoint::PrefixCheckpoints;

pub(in crate::engine) struct MixedMixerExecution {
    template: CudaSharedRoutedModelTemplate,
    caches: Vec<Option<crate::PagedKvCache>>,
    prefill_chunk_tokens: usize,
    sessions: HashMap<Uuid, CudaSharedRoutedModelSession>,
    decode_batches: HashMap<usize, CudaSharedRoutedDecodeBatch>,
    prefill_batches: HashMap<(usize, usize), CudaSharedRoutedPrefillBatch>,
    checkpoints: PrefixCheckpoints,
}

impl MixedMixerExecution {
    pub(in crate::engine) fn new(
        template: CudaSharedRoutedModelTemplate,
        prefill_chunk_tokens: usize,
    ) -> Result<Self> {
        if prefill_chunk_tokens == 0 {
            return Err(Error::InvalidDecoderKernel("shared-routed prefill chunk is empty"));
        }
        let caches = template.allocate_shared_kv()?;
        Ok(Self {
            template,
            caches,
            prefill_chunk_tokens,
            sessions: HashMap::new(),
            decode_batches: HashMap::new(),
            prefill_batches: HashMap::new(),
            checkpoints: PrefixCheckpoints::new(128),
        })
    }

    fn checkpoint_prefix(&mut self, request: &runtime::backend::PrefillRequest) -> Result<()> {
        let session = required(&mut self.sessions, request.session_id)?;
        let position = session.position();
        let declared = request.cache_checkpoints.binary_search(&position).is_ok();
        if !declared && request.terminal_cache_checkpoint() != Some(position) {
            return Ok(());
        }
        let checkpoint = session.checkpoint()?;
        let bytes = checkpoint.bytes();
        self.checkpoints
            .insert(&request.model.id, &request.prompt_tokens, position, checkpoint);
        tracing::debug!(
            backend = "cuda",
            session = %request.session_id,
            prefix_checkpoint_tokens = position,
            checkpoint_bytes = bytes,
            "retained CUDA mixed-mixer prefix checkpoint"
        );
        Ok(())
    }
}

impl GenerationExecution for MixedMixerExecution {
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
        let Some(checkpoint) =
            self.checkpoints
                .lookup(&request.model.id, &request.prompt_tokens, minimum, maximum)
        else {
            return Ok(None);
        };
        let mut session = self.template.instantiate_with_caches(&self.caches)?;
        session.restore_checkpoint(checkpoint)?;
        let position = session.position();
        self.sessions.insert(request.session_id, session);
        tracing::debug!(
            backend = "cuda",
            session = %request.session_id,
            prefix_checkpoint_tokens = position,
            "restored CUDA mixed-mixer prefix checkpoint"
        );
        Ok(Some(position))
    }

    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining.min(self.prefill_chunk_tokens)
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
        let session = required(&mut self.sessions, request.session_id)?;
        session.prefill(request.session_id, tokens, table)?;
        self.checkpoint_prefix(request)?;
        if final_chunk {
            let session = required(&mut self.sessions, request.session_id)?;
            generation_output(backend, session, request.sampling_logits).map(Some)
        } else {
            Ok(None)
        }
    }

    fn prefill_batch_chunk(
        &mut self,
        backend: &CudaBackend,
        chunks: &[super::super::PrefillChunk<'_>],
    ) -> Result<Vec<Option<Output>>> {
        batch::prefill(self, backend, chunks)
    }

    fn decode(
        &mut self,
        backend: &CudaBackend,
        request: &runtime::backend::DecodeRequest,
        _use_device_token: bool,
    ) -> Result<Output> {
        let session = required(&mut self.sessions, request.session_id)?;
        session.decode(request.session_id, request.token_id, &request.block_table)?;
        generation_output(backend, session, request.sampling_logits)
    }

    fn decode_batch(
        &mut self,
        backend: &CudaBackend,
        sequences: &[runtime::backend::DecodeSequence],
    ) -> Result<Option<Vec<Output>>> {
        batch::decode(self, backend, sequences)
    }

    fn clear_sessions(&mut self) {
        self.sessions.clear();
        self.decode_batches.clear();
        self.prefill_batches.clear();
        self.checkpoints.clear();
    }

    fn release_session(&mut self, session: Uuid) {
        self.sessions.remove(&session);
    }

    fn prefill_spatial_vision(
        &mut self,
        backend: &CudaBackend,
        input: SpatialVisionPrefill<'_>,
    ) -> Result<Output> {
        let mut session = self.template.instantiate_with_caches(&self.caches)?;
        session.prefill_vision(
            input.session,
            input.tokens,
            input.positions,
            input.table,
            input.image_span,
            input.image,
            input.position_delta,
        )?;
        let output = generation_output(backend, &mut session, input.sampling)?;
        self.sessions.insert(input.session, session);
        Ok(output)
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
