use uuid::Uuid;

use super::{GenerationExecution, PooledVisionPrefill};
use crate::{
    CudaBackend, CudaMoeModelSession, Result,
    engine::execution::{Output, generation_output},
};

pub(in crate::engine) struct GraphExecution {
    session: CudaMoeModelSession,
}

impl GraphExecution {
    pub(in crate::engine) const fn new(session: CudaMoeModelSession) -> Self {
        Self { session }
    }
}

impl GenerationExecution for GraphExecution {
    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        self.session.prefill_chunk_len(remaining)
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
        self.session.prefill_chunk(request.session_id, tokens, offset, table)?;
        if !final_chunk {
            return Ok(None);
        }
        self.session.finish_prefill(tokens.len(), request.sampling_logits)?;
        generation_output(backend, &mut self.session, request.sampling_logits).map(Some)
    }

    fn decode(
        &mut self,
        backend: &CudaBackend,
        request: &runtime::backend::DecodeRequest,
        use_device_token: bool,
    ) -> Result<Output> {
        if use_device_token {
            self.session.decode_sampled_for_sampling(
                request.session_id,
                &request.block_table,
                request.sampling_logits,
            )?;
        } else {
            self.session.decode_for_sampling(
                request.session_id,
                request.token_id,
                &request.block_table,
                request.sampling_logits,
            )?;
        }
        generation_output(backend, &mut self.session, request.sampling_logits)
    }

    fn clear_sessions(&mut self) {}

    fn release_session(&mut self, _session: Uuid) {}

    fn prefill_pooled_vision(
        &mut self,
        backend: &CudaBackend,
        input: PooledVisionPrefill<'_>,
    ) -> Result<Output> {
        self.session.prefill_vision_for_sampling(
            input.session,
            input.tokens,
            input.image,
            input.image_span.0,
            input.image_span.1,
            input.bidirectional,
            input.table,
            input.sampling,
        )?;
        generation_output(backend, &mut self.session, input.sampling)
    }
}
