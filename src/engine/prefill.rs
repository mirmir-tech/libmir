use runtime::{
    Result as RuntimeResult,
    backend::{ModelHandle, PrefillOutput, PrefillRequest, SamplingLogits},
    kv::BlockTable,
    progress::ProgressEvent,
};
use uuid::Uuid;

use super::{Engine, EngineInner};

impl Engine {
    #[allow(clippy::too_many_arguments)]
    /// Prefills one session from token identifiers and its allocated cache
    /// blocks.
    pub fn prefill_tokens_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt_tokens: &[u32],
        block_table: &BlockTable,
        cached_tokens: usize,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        #[cfg(not(feature = "cuda"))]
        let _ = cached_tokens;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (
            &model, session_id, &prompt_tokens, &block_table, cached_tokens, sampling, &progress,
        );
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.prefill_with_progress(
                &PrefillRequest {
                    model: model.clone(),
                    session_id,
                    prompt_tokens: prompt_tokens.to_vec(),
                    block_table: block_table.clone(),
                    cached_tokens,
                    sampling_logits: sampling,
                },
                progress,
            )?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let mut mapped = |event| progress(super::metal_progress(event));
                metal.prefill_tokens_with_progress(
                    model, session_id, prompt_tokens, block_table, sampling, &mut mapped,
                )
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }

    pub(crate) fn prefill_request_with_progress(
        &self,
        request: &PrefillRequest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        self.prefill_tokens_with_progress(
            &request.model,
            request.session_id,
            &request.prompt_tokens,
            &request.block_table,
            request.cached_tokens,
            request.sampling_logits,
            progress,
        )
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn prefill_requests_with_progress(
        &self,
        requests: &[PrefillRequest],
        token_budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> RuntimeResult<Vec<PrefillOutput>> {
        #[cfg(not(feature = "cuda"))]
        let _ = token_budget;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&requests, &progress);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                Ok(cuda.prefill_batch_with_progress(requests, token_budget, progress)?)
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => requests
                .iter()
                .enumerate()
                .map(|(row, request)| {
                    let mut mapped = |event| progress(row, event);
                    self.prefill_request_with_progress(request, &mut mapped)
                })
                .collect(),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }
}
