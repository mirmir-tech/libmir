use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{
        DecodeOutput, DecodeTimings, ModelHandle, PrefillOutput, PrefillRequest, SamplingLogits,
        TokenEvent,
    },
    kv::BlockTable,
};
use uuid::Uuid;

use super::MetalBackend;
use crate::{
    MetalProgressEvent,
    native::{error::Result, model::LoadedModel, output, prefill::timing::PrefillTiming},
};

impl MetalBackend {
    pub fn prefill_request_with_progress(
        &self,
        request: &PrefillRequest,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        Ok(self.prefill_request_inner(request, progress)?)
    }

    pub fn decode_token(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        token_id: u32,
        block_table: &BlockTable,
        sampling_logits: SamplingLogits,
    ) -> RuntimeResult<DecodeOutput> {
        Ok(self.decode_token_inner(model, session_id, token_id, block_table, sampling_logits)?)
    }

    pub(super) fn prefill_request_inner(
        &self,
        request: &PrefillRequest,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<PrefillOutput> {
        let started = Instant::now();
        let lookup = request.model.id.clone();
        let request = request.clone();
        let execution_sampling = execution_sampling(
            request.sampling_logits.clone(),
            self.config.fusion.device_token_pipeline.enabled(),
        );
        self.with_model_progress(
            &lookup,
            move |loaded, worker_progress| {
                execute_prefill(loaded, &request, execution_sampling, started, worker_progress)
            },
            progress,
        )
    }

    pub(super) fn decode_token_inner(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        token_id: u32,
        block_table: &BlockTable,
        sampling_logits: SamplingLogits,
    ) -> Result<DecodeOutput> {
        let started = Instant::now();
        let lookup = model.id.clone();
        let model_id = lookup.clone();
        let table = block_table.clone();
        let profile = self.profile_decode.load(std::sync::atomic::Ordering::Relaxed);
        let execution_sampling = execution_sampling(
            sampling_logits.clone(),
            self.config.fusion.device_token_pipeline.enabled(),
        );
        self.with_model(&lookup, move |loaded| {
            execute_decode(
                loaded,
                &model_id,
                session_id,
                token_id,
                &table,
                sampling_logits,
                execution_sampling,
                profile,
                started,
            )
        })
    }
}

fn execute_prefill(
    loaded: &mut LoadedModel,
    request: &PrefillRequest,
    execution_sampling: SamplingLogits,
    started: Instant,
    progress: &mut dyn FnMut(MetalProgressEvent),
) -> Result<PrefillOutput> {
    let executing = Instant::now();
    let mut timing = PrefillTiming::new(started);
    let native = loaded.prefill_with_budget(
        request.session_id,
        &request.prompt_tokens,
        &request.cache_checkpoints,
        execution_sampling,
        request.block_table.block_size(),
        request.generation_tokens,
        progress,
    )?;
    timing.record(executing.elapsed());
    let output = super::prefill_output::materialize_prefill_parts(
        loaded,
        &request.model.id,
        request.session_id,
        &request.prompt_tokens,
        request.block_table.blocks().len(),
        request.sampling_logits.clone(),
        native,
        timing,
    );
    loaded.finish_sessions([request.session_id], output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_decode(
    loaded: &mut LoadedModel,
    model_id: &str,
    session_id: Uuid,
    token_id: u32,
    block_table: &BlockTable,
    sampling_logits: SamplingLogits,
    execution_sampling: SamplingLogits,
    profile: bool,
    started: Instant,
) -> Result<DecodeOutput> {
    let executing = Instant::now();
    let native = loaded.decode(session_id, token_id, execution_sampling.clone())?;
    let output = output::materialize(loaded, native, sampling_logits);
    let output = loaded.finish_decode(
        &[crate::native::model::DecodeInput {
            session: session_id,
            token: token_id,
            sampling: execution_sampling,
        }],
        output,
    )?;
    let cached_tokens = loaded.session_cached_tokens(session_id)?;
    let finished = Instant::now();
    let timings = super::decode_timing::finish(started, executing, finished);
    let elapsed = finished.duration_since(started);
    let trace = decode_trace(profile, block_table, cached_tokens, elapsed);
    tracing::trace!(
        model_id,
        session_id = %session_id,
        token_id,
        cached_tokens,
        "native MLX decode completed"
    );
    Ok(DecodeOutput {
        event: TokenEvent {
            token_id: output.next_token,
            text: trace,
            finished: false,
        },
        logits: output.logits,
        candidates: output.candidates,
        timings: profile.then_some(DecodeTimings { batch_rows: 1, ..timings }),
    })
}

fn device_pipeline(sampling: SamplingLogits, enabled: bool) -> bool {
    matches!(
        sampling,
        SamplingLogits::None | SamplingLogits::SampleTopK { .. } | SamplingLogits::Sample { .. }
    ) && enabled
}

pub(super) fn execution_sampling(
    sampling: SamplingLogits,
    device_pipeline_enabled: bool,
) -> SamplingLogits {
    if matches!(sampling, SamplingLogits::Masked { .. }) {
        return sampling;
    }
    if device_pipeline(sampling.clone(), device_pipeline_enabled) {
        sampling
    } else {
        SamplingLogits::Full
    }
}

fn decode_trace(
    profile: bool,
    block_table: &BlockTable,
    cached_tokens: usize,
    elapsed: std::time::Duration,
) -> String {
    if !profile {
        return "native decode on explicit MLX GPU stream".into();
    }
    format!(
        "decode.stage_profile: native stream, {} runtime KV blocks, {} cached tokens, {:.3}ms",
        block_table.blocks().len(),
        cached_tokens,
        elapsed.as_secs_f64() * 1000.0
    )
}
