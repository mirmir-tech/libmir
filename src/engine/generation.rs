use runtime::{
    backend::{
        DecodeBatchRequest, DecodeOutput, DecodeSequence, ModelHandle, PrefillOutput,
        PrefillRequest,
    },
    kv::CacheConfig,
    progress::ProgressEvent,
};

use super::{Engine, EngineInner};
use crate::Result;

const METAL_COMPLETION_ROUND_ROWS: usize = 4;
const METAL_MAX_PREFILL_WAVE_ROWS: usize = 2;

pub enum EnginePrefillBatch {
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaPrefillBatch),
    #[cfg(feature = "metal")]
    Metal(metal::MetalPrefillBatch),
}

pub struct EngineGenerationStepOutput {
    pub decode: Vec<DecodeOutput>,
    pub prefill: Result<bool>,
}

#[derive(Clone, Copy)]
pub struct PrefillExecutionProfile {
    pub chunk_tokens: usize,
    pub completion_round_tokens: usize,
    pub max_prefill_wave_rows: usize,
    pub block_tokens: usize,
    pub resident_token_slots: usize,
    pub limit_deep_prefill_waves: bool,
    pub cached_prefix_replay_tokens: Option<usize>,
    pub cached_prefix_checkpoint_replay_tokens: Option<usize>,
    pub cached_prefix_completion_slack_tokens: usize,
    pub defer_new_decode: bool,
    pub collect_long_prefill_window: bool,
}

impl Engine {
    #[cfg_attr(
        not(feature = "metal"),
        allow(
            clippy::unnecessary_wraps,
            reason = "the Metal implementation performs a fallible model lookup"
        )
    )]
    pub(crate) fn generation_prefill_profile(
        &self,
        model: &ModelHandle,
        max_batch_tokens: usize,
        cache: CacheConfig,
    ) -> Result<PrefillExecutionProfile> {
        #[cfg(not(feature = "metal"))]
        let _ = model;
        #[cfg(not(feature = "cuda"))]
        let _ = max_batch_tokens;
        let resident_token_slots =
            cache.block_size.saturating_mul(cache.block_count as usize).max(1);
        let chunk_tokens = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.prefill_chunk_tokens(),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.prefill_chunk_tokens(model)?,
        };
        let completion_round_tokens = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => max_batch_tokens,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => max_batch_tokens.div_ceil(METAL_COMPLETION_ROUND_ROWS),
        };
        let max_prefill_wave_rows = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => usize::MAX,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => METAL_MAX_PREFILL_WAVE_ROWS,
        };
        let limit_deep_prefill_waves = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => true,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => true,
        };
        let cached_prefix_admission = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.paged_prefix_admission(&model.id)?,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Some((0, 0, 0)),
        };
        let cached_prefix_replay_tokens =
            cached_prefix_admission.map(|(fallback_tokens, _, _)| fallback_tokens);
        let cached_prefix_checkpoint_replay_tokens =
            cached_prefix_admission.map(|(_, checkpoint_tokens, _)| checkpoint_tokens);
        let cached_prefix_completion_slack_tokens =
            cached_prefix_admission.map_or(0, |(_, _, slack_tokens)| slack_tokens);
        let defer_new_decode = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => false,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => true,
        };
        let collect_long_prefill_window = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => false,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => true,
        };
        Ok(PrefillExecutionProfile {
            chunk_tokens: chunk_tokens.max(1),
            completion_round_tokens: completion_round_tokens.max(1),
            max_prefill_wave_rows,
            block_tokens: cache.block_size.max(1),
            resident_token_slots,
            limit_deep_prefill_waves,
            cached_prefix_replay_tokens,
            cached_prefix_checkpoint_replay_tokens,
            cached_prefix_completion_slack_tokens,
            defer_new_decode,
            collect_long_prefill_window,
        })
    }

    pub(crate) fn prepare_generation_prefill(
        &self,
        requests: &[PrefillRequest],
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<EnginePrefillBatch> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                Ok(EnginePrefillBatch::Cuda(cuda.prepare_prefill_batch(requests, progress)?))
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let mut mapped = |row, event| progress(row, metal_progress(event));
                Ok(EnginePrefillBatch::Metal(metal.prepare_prefill_batch(requests, &mut mapped)?))
            },
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn execute_generation_step(
        &self,
        model: &ModelHandle,
        sequences: Vec<DecodeSequence>,
        prefill: Option<&mut EnginePrefillBatch>,
        prefill_budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<EngineGenerationStepOutput> {
        let request = if sequences.is_empty() {
            None
        } else {
            Some(DecodeBatchRequest::new(model.clone(), sequences)?)
        };
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let batch = match prefill {
                    Some(EnginePrefillBatch::Cuda(batch)) => Some(batch),
                    None => None,
                    #[cfg(feature = "metal")]
                    Some(EnginePrefillBatch::Metal(_)) => return Err(batch_backend_mismatch()),
                };
                let output = cuda.execute_generation_step(
                    request.as_ref(),
                    batch,
                    prefill_budget,
                    progress,
                )?;
                Ok(EngineGenerationStepOutput {
                    decode: output.decode,
                    prefill: match output.prefill {
                        Ok(prefill) => Ok(prefill),
                        Err(error) => Err(error.into()),
                    },
                })
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let batch = match prefill {
                    Some(EnginePrefillBatch::Metal(batch)) => Some(batch),
                    None => None,
                    #[cfg(feature = "cuda")]
                    Some(EnginePrefillBatch::Cuda(_)) => return Err(batch_backend_mismatch()),
                };
                let mut mapped = |row, event| progress(row, metal_progress(event));
                let output = metal.execute_generation_step(
                    request.as_ref(),
                    batch.as_deref(),
                    prefill_budget,
                    &mut mapped,
                )?;
                Ok(EngineGenerationStepOutput {
                    decode: output.decode,
                    prefill: match output.prefill {
                        Ok(prefill) => Ok(prefill),
                        Err(error) => Err(error.into()),
                    },
                })
            },
        }
    }

    pub(crate) fn finish_generation_prefill(
        &self,
        batch: EnginePrefillBatch,
    ) -> Result<Vec<PrefillOutput>> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => match batch {
                EnginePrefillBatch::Cuda(batch) => Ok(cuda.finish_prefill_batch(batch)?),
                #[cfg(feature = "metal")]
                EnginePrefillBatch::Metal(_) => Err(batch_backend_mismatch()),
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => match batch {
                EnginePrefillBatch::Metal(batch) => Ok(metal.finish_prefill_batch(batch)?),
                #[cfg(feature = "cuda")]
                EnginePrefillBatch::Cuda(_) => Err(batch_backend_mismatch()),
            },
        }
    }
}

#[cfg(feature = "metal")]
fn metal_progress(event: metal::MetalProgressEvent) -> ProgressEvent {
    ProgressEvent {
        stage: match event.stage {
            metal::MetalProgressStage::LoadWeights => runtime::progress::ProgressStage::LoadWeights,
            metal::MetalProgressStage::PrefillTokens => {
                runtime::progress::ProgressStage::PrefillTokens
            },
        },
        current: event.current,
        total: event.total,
        unit: match event.unit {
            metal::MetalProgressUnit::Byte => runtime::progress::ProgressUnit::Byte,
            metal::MetalProgressUnit::Token => runtime::progress::ProgressUnit::Token,
        },
        detail: event.detail,
    }
}

#[cfg(all(feature = "cuda", feature = "metal"))]
fn batch_backend_mismatch() -> crate::Error {
    runtime::RuntimeError::Backend("generation prefill batch targets another backend".into()).into()
}
