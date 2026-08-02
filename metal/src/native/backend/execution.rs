use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{
        DecodeOutput, DecodeTimings, ModelHandle, PrefillOutput, SamplingLogits, TokenEvent,
    },
    kv::BlockTable,
};
use uuid::Uuid;

use super::MetalBackend;
use crate::{
    MetalProgressEvent,
    native::{error::Result, model::LoadedModel, output},
};

impl MetalBackend {
    pub fn prefill_tokens_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt_tokens: &[u32],
        block_table: &BlockTable,
        sampling_logits: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        Ok(self.prefill_tokens_inner(
            model, session_id, prompt_tokens, block_table, sampling_logits, progress,
        )?)
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

    pub(super) fn prefill_tokens_inner(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt_tokens: &[u32],
        block_table: &BlockTable,
        sampling_logits: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<PrefillOutput> {
        let started = Instant::now();
        let lookup = model.id.clone();
        let model_id = lookup.clone();
        let tokens = prompt_tokens.to_vec();
        let blocks = block_table.blocks().len();
        let block_size = block_table.block_size();
        let execution_sampling =
            execution_sampling(sampling_logits, self.config.fusion.device_token_pipeline.enabled());
        self.with_model_progress(
            &lookup,
            move |loaded, worker_progress| {
                execute_prefill(
                    loaded,
                    &model_id,
                    session_id,
                    &tokens,
                    blocks,
                    block_size,
                    sampling_logits,
                    execution_sampling,
                    started,
                    worker_progress,
                )
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
        let execution_sampling =
            execution_sampling(sampling_logits, self.config.fusion.device_token_pipeline.enabled());
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

#[allow(clippy::too_many_arguments)]
fn execute_prefill(
    loaded: &mut LoadedModel,
    model_id: &str,
    session_id: Uuid,
    prompt_tokens: &[u32],
    blocks: usize,
    block_size: Option<usize>,
    sampling_logits: SamplingLogits,
    execution_sampling: SamplingLogits,
    started: Instant,
    progress: &mut dyn FnMut(MetalProgressEvent),
) -> Result<PrefillOutput> {
    let native =
        loaded.prefill(session_id, prompt_tokens, execution_sampling, block_size, progress)?;
    super::prefill_output::materialize_prefill_parts(
        loaded, model_id, session_id, prompt_tokens, blocks, sampling_logits, native, started,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_decode(
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
    let native = loaded.decode(session_id, token_id, execution_sampling)?;
    let output = output::materialize(loaded, native, sampling_logits)?;
    let cached_tokens = loaded.session_cached_tokens(session_id)?;
    let elapsed = started.elapsed();
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
        timings: profile.then(|| DecodeTimings {
            backend_execution: elapsed,
            batch_rows: 1,
            ..DecodeTimings::default()
        }),
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
    if device_pipeline(sampling, device_pipeline_enabled) {
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
