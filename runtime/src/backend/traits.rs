use std::time::Duration;

use async_trait::async_trait;
use foundation::model::ModelManifest;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{DecodeBatchOutput, DecodeBatchRequest};
use crate::{
    error::{Result, RuntimeError},
    kv::BlockTable,
    trace::ModelTrace,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHandle {
    pub id: String,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub session_id: Uuid,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    pub token_id: Option<u32>,
    pub text: String,
    pub finished: bool,
}

#[derive(Debug, Clone)]
pub struct LogitsTrace {
    pub shape: Vec<i32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CandidateLogitsTrace {
    pub token_ids: Vec<u32>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingLogits {
    None,
    Full,
    TopK {
        k: usize,
        vocab_size: usize,
    },
    SampleTopK {
        k: usize,
        vocab_size: usize,
        temperature: f32,
        draw: f32,
    },
    Sample {
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        draw: f32,
    },
}

impl SamplingLogits {
    #[must_use]
    pub const fn requires_history(self) -> bool {
        matches!(self, Self::Full | Self::TopK { .. })
    }
}

impl LogitsTrace {
    #[must_use]
    pub fn finite_count(&self) -> usize {
        self.values.iter().filter(|value| value.is_finite()).count()
    }

    #[must_use]
    pub fn non_finite_count(&self) -> usize {
        self.values.len().saturating_sub(self.finite_count())
    }

    #[must_use]
    pub fn finite_min_max(&self) -> Option<(f32, f32)> {
        let mut finite = self.values.iter().copied().filter(|value| value.is_finite());
        let first = finite.next()?;
        Some(finite.fold((first, first), |(min, max), value| (min.min(value), max.max(value))))
    }
}

impl CandidateLogitsTrace {
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_ids.len().min(self.scores.len())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct PrefillRequest {
    pub model: ModelHandle,
    pub session_id: Uuid,
    pub prompt_tokens: Vec<u32>,
    pub cache_checkpoints: Vec<usize>,
    pub block_table: BlockTable,
    pub cached_tokens: usize,
    pub sampling_logits: SamplingLogits,
}

impl PrefillRequest {
    /// Last complete K/V block whose recurrent state can be reused safely.
    #[must_use]
    pub fn terminal_cache_checkpoint(&self) -> Option<usize> {
        let block = self.block_table.block_size().filter(|block| *block > 0)?;
        let before_tail = self.prompt_tokens.len().checked_sub(block)?;
        let checkpoint = before_tail / block * block;
        (checkpoint > 0).then_some(checkpoint)
    }
}

#[derive(Debug, Clone)]
pub struct PrefillOutput {
    pub accepted_tokens: usize,
    pub next_token: Option<u32>,
    pub trace: Option<String>,
    pub logits: Option<LogitsTrace>,
    pub candidates: Option<CandidateLogitsTrace>,
}

#[derive(Debug, Clone)]
pub struct DecodeRequest {
    pub model: ModelHandle,
    pub session_id: Uuid,
    pub token_id: u32,
    pub block_table: BlockTable,
    pub sampling_logits: SamplingLogits,
}

#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub event: TokenEvent,
    pub logits: Option<LogitsTrace>,
    pub candidates: Option<CandidateLogitsTrace>,
    pub timings: Option<DecodeTimings>,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct DecodeTimings {
    pub scheduler_queue: Duration,
    pub backend_wait: Duration,
    pub backend_execution: Duration,
    pub device_execution: Option<Duration>,
    pub batch_rows: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub model: ModelHandle,
    pub inputs: Vec<Vec<u32>>,
    pub dimensions: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct SequenceScoringRequest {
    pub model: ModelHandle,
    pub pairs: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct SequenceScoringOutput {
    pub scores: Vec<f32>,
    pub prompt_tokens: usize,
}

#[async_trait]
pub trait Backend: Send + Sync {
    #[must_use]
    fn info(&self) -> super::BackendInfo;
    async fn load_model(&self, manifest: &ModelManifest) -> Result<ModelHandle>;
    async fn model_trace(&self, model: &ModelHandle) -> Result<ModelTrace> {
        Err(RuntimeError::BackendUnavailable(format!(
            "model trace is not implemented for {}",
            model.backend
        )))
    }
    async fn prefill(&self, request: PrefillRequest) -> Result<PrefillOutput> {
        Err(RuntimeError::BackendUnavailable(format!(
            "prefill is not implemented for {}",
            request.model.backend
        )))
    }
    async fn decode(&self, request: DecodeRequest) -> Result<DecodeOutput> {
        Err(RuntimeError::BackendUnavailable(format!(
            "decode is not implemented for {}",
            request.model.backend
        )))
    }
    async fn decode_batch(&self, request: DecodeBatchRequest) -> Result<DecodeBatchOutput> {
        Err(RuntimeError::BackendUnavailable(format!(
            "batched decode is not implemented for {}",
            request.model().backend
        )))
    }
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingOutput> {
        Err(RuntimeError::BackendUnavailable(format!(
            "embedding is not implemented for {}",
            request.model.backend
        )))
    }
    async fn score_sequences(
        &self,
        request: SequenceScoringRequest,
    ) -> Result<SequenceScoringOutput> {
        Err(RuntimeError::BackendUnavailable(format!(
            "sequence scoring is not implemented for {}",
            request.model.backend
        )))
    }
    async fn generate(
        &self,
        model: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Vec<TokenEvent>>;
}
