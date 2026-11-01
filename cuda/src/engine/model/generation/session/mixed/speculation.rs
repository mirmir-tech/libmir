//! Speculative decoding with the model's prediction module: scalar drafting
//! prefill chunks and verification steps. Packed steps draft in `drafting`;
//! these paths take what they cannot.

use runtime::{
    backend::{DecodeRequest, PrefillRequest},
    kv::BlockTable,
};

use super::{
    super::super::super::{GenerationExecution, PrefillChunk},
    MixedMixerExecution, required,
};
use crate::{
    CudaBackend, Error, Result,
    engine::execution::{Output, generation_output},
};

/// Drafting policy of a model with a prediction module; each session keeps
/// its drafts on the device.
#[derive(Clone, Copy, Debug)]
pub(super) struct Drafting {
    draft_tokens: usize,
}

impl Drafting {
    pub(super) const fn new(speculation: runtime::speculation::Speculation) -> Self {
        Self {
            draft_tokens: speculation.draft_tokens.get(),
        }
    }

    pub(super) const fn draft_tokens(self) -> usize {
        self.draft_tokens
    }
}

/// Drafts a decode step at `position` can verify and draft again within the
/// blocks of `table`.
pub(super) fn room(table: &BlockTable, position: usize) -> Result<usize> {
    Ok(capacity(table)?.saturating_sub(position + 1) / 2)
}

impl MixedMixerExecution {
    /// Prefills one chunk and runs the prediction module over its rows. The
    /// last row of the last chunk is followed by the sampled token, from
    /// which the first drafts follow.
    pub(super) fn prefill_drafting_chunk(
        &mut self,
        backend: &CudaBackend,
        request: &PrefillRequest,
        tokens: &[u32],
        offset: usize,
        table: &BlockTable,
        final_chunk: bool,
    ) -> Result<Option<Output>> {
        let session_id = request.session_id;
        let rows = tokens.len();
        let session = required(&mut self.sessions, session_id)?;
        if !final_chunk {
            let next = request
                .prompt_tokens
                .get(offset + 1..=offset + rows)
                .ok_or(Error::InvalidDecoderKernel("prefill chunk exceeds the prompt"))?;
            session.prefill_drafting(session_id, tokens, next, table)?;
            self.checkpoint_prefix(request)?;
            return Ok(None);
        }
        session.prefill_rows(session_id, tokens, table)?;
        self.checkpoint_prefix(request)?;
        let session = required(&mut self.sessions, session_id)?;
        let output = generation_output(backend, session, request.sampling_logits.clone())?;
        let (Some(token), Some(drafting)) = (output.token, self.drafting) else {
            return Ok(Some(output));
        };
        let next = tokens[1..].iter().copied().chain([token]).collect::<Vec<_>>();
        session.predict(session_id, rows, &next, table)?;
        let position = session.position() - 1;
        let count = drafting.draft_tokens.min(capacity(table)?.saturating_sub(position + 1));
        session.draft(session_id, rows, rows - 1, position, count, table)?;
        Ok(Some(output))
    }

    /// Prefills batched chunks in one drafting pass, or one at a time where
    /// the pass does not apply.
    pub(super) fn prefill_drafting_chunks(
        &mut self,
        backend: &CudaBackend,
        chunks: &[PrefillChunk<'_>],
    ) -> Result<Vec<Option<Output>>> {
        if let Some(outputs) = super::drafting::execute(self, chunks, &[])? {
            return Ok(outputs.prefill);
        }
        chunks
            .iter()
            .map(|chunk| {
                self.prefill_chunk(
                    backend,
                    chunk.request,
                    chunk.tokens,
                    chunk.offset,
                    chunk.table,
                    chunk.final_chunk,
                )
            })
            .collect()
    }

    /// Feeds the request's token and verifies the session's drafts after it.
    pub(super) fn decode_speculative(&mut self, request: &DecodeRequest) -> Result<Output> {
        let drafting = self.drafting.ok_or(Error::InvalidDecoderKernel("model does not draft"))?;
        let session = required(&mut self.sessions, request.session_id)?;
        // This step verifies its drafts and drafts as many after them.
        let room = room(&request.block_table, session.position())?;
        let step = session.decode_speculative(
            request.session_id,
            request.token_id,
            drafting.draft_tokens.min(room),
            &request.sampling_logits,
            &request.speculation.clone().unwrap_or_default(),
            &request.block_table,
        )?;
        Ok(Output {
            accepted: step.accepted,
            token: Some(step.next),
            logits: None,
        })
    }

    /// Drops the drafts of sessions that decode as a plain batch, which
    /// leaves the prediction module's K/V behind.
    pub(super) fn forget_drafts(&mut self, sequences: &[runtime::backend::DecodeSequence]) {
        for sequence in sequences {
            if let Some(session) = self.sessions.get_mut(&sequence.session_id) {
                session.forget_drafts();
            }
        }
    }
}

fn capacity(table: &BlockTable) -> Result<usize> {
    let block_size = table.block_size().ok_or(Error::InvalidPagedKv("table has no block size"))?;
    Ok(table.capacity(block_size))
}
