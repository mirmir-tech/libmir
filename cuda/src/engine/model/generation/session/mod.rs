use std::collections::HashMap;

use uuid::Uuid;

use super::{GenerationExecution, PrefillChunk, SpatialVisionPrefill};
use crate::{
    CudaBackend, CudaClampedRoutedModelSession, CudaClampedRoutedModelTemplate,
    CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, Error, Result,
    engine::execution::{Output, generation_output},
};

mod batch;

pub(in crate::engine) struct MixedMixerExecution {
    template: CudaSharedRoutedModelTemplate,
    sessions: HashMap<Uuid, CudaSharedRoutedModelSession>,
}

pub(in crate::engine) struct SinkAttentionExecution {
    session: CudaClampedRoutedModelSession,
    prefix_replay_tokens: Option<usize>,
    prefix_checkpoint_block_tokens: usize,
}

impl MixedMixerExecution {
    pub(in crate::engine) fn new(template: CudaSharedRoutedModelTemplate) -> Self {
        Self { template, sessions: HashMap::new() }
    }
}

impl SinkAttentionExecution {
    pub(in crate::engine) fn new(template: &CudaClampedRoutedModelTemplate) -> Result<Self> {
        let caches = template.allocate_shared_kv()?;
        let session = template.instantiate_with_caches(&caches)?;
        Ok(Self {
            session,
            prefix_replay_tokens: template.prefix_replay_tokens(),
            prefix_checkpoint_block_tokens: template.prefix_checkpoint_block_tokens(),
        })
    }
}

impl GenerationExecution for MixedMixerExecution {
    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining
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
            self.sessions.insert(request.session_id, self.template.instantiate()?);
        }
        let session = required(&mut self.sessions, request.session_id)?;
        session.prefill(request.session_id, tokens, table)?;
        if final_chunk {
            generation_output(backend, session, request.sampling_logits).map(Some)
        } else {
            Ok(None)
        }
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

    fn clear_sessions(&mut self) {
        self.sessions.clear();
    }

    fn release_session(&mut self, session: Uuid) {
        self.sessions.remove(&session);
    }

    fn prefill_spatial_vision(
        &mut self,
        backend: &CudaBackend,
        input: SpatialVisionPrefill<'_>,
    ) -> Result<Output> {
        let mut session = self.template.instantiate()?;
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

impl GenerationExecution for SinkAttentionExecution {
    fn prefix_replay_tokens(&self) -> Option<usize> {
        self.prefix_replay_tokens
    }

    fn prefix_admission(&self) -> Option<(usize, usize, usize)> {
        self.prefix_replay_tokens
            .map(|fallback| (fallback, 0, self.prefix_checkpoint_block_tokens))
    }

    fn restore_prefix(
        &mut self,
        request: &runtime::backend::PrefillRequest,
        minimum: usize,
        maximum: usize,
    ) -> Result<Option<usize>> {
        self.session.restore_prefix(
            request.session_id,
            &request.model.id,
            &request.prompt_tokens,
            minimum,
            maximum,
        )
    }

    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining
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
        self.session.begin_chunk(request.session_id, offset)?;
        self.session.prefill_with_sampling(
            request.session_id,
            tokens,
            table,
            request.cached_tokens,
            request.sampling_logits,
        )?;
        self.session.checkpoint_prefix(
            request.session_id,
            &request.model.id,
            &request.prompt_tokens,
            offset + tokens.len(),
        )?;
        if final_chunk {
            generation_output(backend, &mut self.session, request.sampling_logits).map(Some)
        } else {
            Ok(None)
        }
    }

    fn prefill_batch_chunk(
        &mut self,
        backend: &CudaBackend,
        chunks: &[PrefillChunk<'_>],
    ) -> Result<Vec<Option<Output>>> {
        let tokens =
            chunks.iter().flat_map(|chunk| chunk.tokens.iter().copied()).collect::<Vec<_>>();
        let sessions = chunks.iter().map(|chunk| chunk.request.session_id).collect::<Vec<_>>();
        let tables = chunks.iter().map(|chunk| chunk.table).collect::<Vec<_>>();
        let starts = chunks.iter().map(|chunk| chunk.offset).collect::<Vec<_>>();
        let counts = chunks.iter().map(|chunk| chunk.tokens.len()).collect::<Vec<_>>();
        let write_starts =
            chunks.iter().map(|chunk| chunk.request.cached_tokens).collect::<Vec<_>>();
        for (session, start) in sessions.iter().zip(&starts) {
            self.session.begin_chunk(*session, *start)?;
        }
        self.session
            .prefill_packed_chunk(&sessions, &tokens, &tables, &starts, &counts, &write_starts)?;
        for chunk in chunks {
            self.session.checkpoint_prefix(
                chunk.request.session_id,
                &chunk.request.model.id,
                &chunk.request.prompt_tokens,
                chunk.offset + chunk.tokens.len(),
            )?;
        }
        let total = tokens.len();
        let mut packed = 0;
        chunks
            .iter()
            .map(|chunk| {
                packed += chunk.tokens.len();
                if !chunk.final_chunk {
                    return Ok(None);
                }
                self.session.finish_packed_prefill_row(
                    packed - 1,
                    total,
                    chunk.request.sampling_logits,
                )?;
                generation_output(backend, &mut self.session, chunk.request.sampling_logits)
                    .map(Some)
            })
            .collect()
    }

    fn decode(
        &mut self,
        backend: &CudaBackend,
        request: &runtime::backend::DecodeRequest,
        _use_device_token: bool,
    ) -> Result<Output> {
        self.session.decode_with_sampling(
            request.session_id,
            request.token_id,
            &request.block_table,
            request.sampling_logits,
        )?;
        generation_output(backend, &mut self.session, request.sampling_logits)
    }

    fn decode_batch(
        &mut self,
        backend: &CudaBackend,
        sequences: &[runtime::backend::DecodeSequence],
    ) -> Result<Option<Vec<Output>>> {
        batch::decode(&mut self.session, backend, sequences)
    }

    fn clear_sessions(&mut self) {
        self.session.clear_sessions();
    }

    fn release_session(&mut self, session: Uuid) {
        self.session.release_session(session);
    }
}

fn required<T>(sessions: &mut HashMap<Uuid, T>, id: Uuid) -> Result<&mut T> {
    sessions
        .get_mut(&id)
        .ok_or_else(|| Error::State("CUDA decoder session is not initialized".into()))
}
