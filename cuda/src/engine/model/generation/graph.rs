use uuid::Uuid;

use super::{GenerationExecution, PooledVisionPrefill, PrefillChunk};
use crate::{
    CudaBackend, CudaMoeModelSession, Result,
    engine::execution::{Output, device_sampling, generation_output},
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
    fn prefix_replay_tokens(&self) -> Option<usize> {
        Some(0)
    }

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

    fn prefill_batch_chunk(
        &mut self,
        backend: &CudaBackend,
        chunks: &[PrefillChunk<'_>],
    ) -> Result<Vec<Option<Output>>> {
        let tokens =
            chunks.iter().flat_map(|chunk| chunk.tokens.iter().copied()).collect::<Vec<_>>();
        let tables = chunks.iter().map(|chunk| chunk.table).collect::<Vec<_>>();
        let starts = chunks.iter().map(|chunk| chunk.offset).collect::<Vec<_>>();
        let counts = chunks.iter().map(|chunk| chunk.tokens.len()).collect::<Vec<_>>();
        self.session.prefill_packed_chunk(&tokens, &tables, &starts, &counts)?;
        let total = tokens.len();
        let (rows, policies) = packed_output_rows(chunks);
        if rows.len() > 1 && policies.iter().copied().all(device_sampling) {
            let tokens = self.session.finish_packed_prefill_rows(&rows, total, &policies)?;
            return device_outputs(chunks, &tokens);
        }
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

fn packed_output_rows(
    chunks: &[PrefillChunk<'_>],
) -> (Vec<usize>, Vec<runtime::backend::SamplingLogits>) {
    let mut packed = 0;
    let mut rows = Vec::new();
    let mut policies = Vec::new();
    for chunk in chunks {
        packed += chunk.tokens.len();
        if chunk.final_chunk {
            rows.push(packed - 1);
            policies.push(chunk.request.sampling_logits);
        }
    }
    (rows, policies)
}

fn device_outputs(chunks: &[PrefillChunk<'_>], tokens: &[u32]) -> Result<Vec<Option<Output>>> {
    let mut selected = tokens.iter().copied();
    let outputs = chunks
        .iter()
        .map(|chunk| {
            if !chunk.final_chunk {
                return Ok(None);
            }
            let token = selected
                .next()
                .ok_or_else(|| crate::Error::InvalidSampling("missing packed token row".into()))?;
            Ok(Some(Output { token: Some(token), logits: None }))
        })
        .collect::<Result<Vec<_>>>()?;
    if selected.next().is_some() {
        return Err(crate::Error::InvalidSampling("extra packed token rows".into()));
    }
    Ok(outputs)
}
