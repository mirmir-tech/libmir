use std::collections::HashMap;

use uuid::Uuid;

use super::{GenerationExecution, SpatialVisionPrefill};
use crate::{
    CudaBackend, CudaClampedRoutedModelSession, CudaClampedRoutedModelTemplate,
    CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, Error, Result,
    engine::execution::{Output, generation_output},
};

pub(in crate::engine) struct MixedMixerExecution {
    template: CudaSharedRoutedModelTemplate,
    sessions: HashMap<Uuid, CudaSharedRoutedModelSession>,
}

pub(in crate::engine) struct SinkAttentionExecution {
    template: CudaClampedRoutedModelTemplate,
    sessions: HashMap<Uuid, CudaClampedRoutedModelSession>,
}

impl MixedMixerExecution {
    pub(in crate::engine) fn new(template: CudaSharedRoutedModelTemplate) -> Self {
        Self { template, sessions: HashMap::new() }
    }
}

impl SinkAttentionExecution {
    pub(in crate::engine) fn new(template: CudaClampedRoutedModelTemplate) -> Self {
        Self { template, sessions: HashMap::new() }
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
    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining.min(256)
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
}

fn required<T>(sessions: &mut HashMap<Uuid, T>, id: Uuid) -> Result<&mut T> {
    sessions
        .get_mut(&id)
        .ok_or_else(|| Error::State("CUDA decoder session is not initialized".into()))
}
