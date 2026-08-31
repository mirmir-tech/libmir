use std::collections::HashMap;

use foundation::model::ModelManifest;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    layout::{DecoderConfig, EncoderConfig, ModelLayout, ModelMetadata, VisionConfig},
    tokenizer::TokenizerInfo,
    weights::TensorReadiness,
};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{
    error::{Error, Result},
    prefix::PrefixCache,
    session::SessionState,
};
use crate::engine::{
    Array, DecoderModel, MemoryStats, PooledVisionTower, SequenceScoringModel,
    SpatialMergeVisionTower, Stream, TextEmbeddingModel,
};

mod batch;
#[allow(clippy::self_named_module_files)]
mod load;
mod memory;
mod prefill;

pub(super) use batch::DecodeInput;
pub(super) use memory::{cache_prefix_checkpoint, cache_prefix_snapshot};
use prefill::prefill_step;

pub(super) const KV_CACHE_STEP: usize = 256;

#[derive(Debug)]
pub(super) struct ModelInfo {
    pub manifest: ModelManifest,
    pub layout: ModelLayout,
    pub metadata: ModelMetadata,
    pub decoder: Option<DecoderConfig>,
    pub encoder: Option<EncoderConfig>,
    pub vision: Option<VisionConfig>,
    pub vision_readiness: Option<TensorReadiness>,
    pub contract: Option<DecoderExecutionContract>,
    pub task_plan: TaskExecutionPlan,
    pub tensor_count: usize,
    pub weight_bytes: u64,
    pub cache_step: usize,
    pub prefill_step: usize,
    pub tokenizer: Option<TokenizerInfo>,
    pub tokenizer_error: Option<String>,
    pub metal_memory: MemoryStats,
}

#[derive(Debug)]
pub(super) struct LoadedModel {
    pub info: ModelInfo,
    pub(super) stream: Stream,
    pub(super) execution: LoadedExecution,
    pub(super) vision_model: Option<LoadedVisionModel>,
    pub(super) prefixes: PrefixCache,
    pub(super) sessions: HashMap<Uuid, SessionState>,
}

#[derive(Debug)]
pub(super) enum LoadedExecution {
    Generation(DecoderModel),
    Embedding(TextEmbeddingModel),
    SequenceScoring(Box<SequenceScoringModel>),
}

impl LoadedExecution {
    pub(super) fn decoder(&self) -> Result<&DecoderModel> {
        match self {
            Self::Generation(model) => Ok(model),
            Self::Embedding(_) | Self::SequenceScoring(_) => {
                Err(Error::UnsupportedModel("loaded task does not support generation".into()))
            },
        }
    }
}

#[derive(Debug)]
pub(super) enum LoadedVisionModel {
    PooledEncoder(PooledVisionTower),
    SpatialMergeEncoder(SpatialMergeVisionTower),
}

#[derive(Debug)]
pub(super) enum NativeOutput {
    Greedy(u32),
    Logits(Array),
}

impl LoadedModel {
    pub fn decode(
        &mut self,
        session: Uuid,
        token: u32,
        sampling: SamplingLogits,
    ) -> Result<NativeOutput> {
        let model = self.execution.decoder()?;
        let stream = &self.stream;
        let state = self.sessions.get_mut(&session).ok_or_else(|| Error::Session {
            model: self.info.manifest.id.clone(),
            session,
        })?;
        if state.pending.is_some() {
            return super::step::decode_pending(model, stream, state, token, sampling);
        }
        let position = state.model_position()?;
        let logits = super::step::forward_token(model, stream, state, token, position, false)?;
        state.position += 1;
        Ok(NativeOutput::Logits(logits))
    }

    pub(super) fn session_cached_tokens(&self, session: Uuid) -> Result<usize> {
        self.sessions.get(&session).map_or_else(
            || {
                Err(Error::Session {
                    model: self.info.manifest.id.clone(),
                    session,
                })
            },
            |state| Ok(state.position),
        )
    }

    pub(super) fn resident_cached_tokens(&self) -> usize {
        self.sessions.values().map(|state| state.position).sum()
    }

    pub(super) fn release_session(&mut self, session: Uuid) -> Result<()> {
        if self.sessions.contains_key(&session) {
            self.flush_decode_graphs()?;
        }
        let _removed = self.sessions.remove(&session);
        let _reclaimed = Self::reclaim_prefill_allocator_cache()?;
        Ok(())
    }

    pub(super) const fn prefix_cache_enabled(&self) -> bool {
        self.prefixes.enabled()
    }

    pub(super) const fn prefix_cache_capacity(&self) -> usize {
        self.prefixes.capacity()
    }

    pub(super) const fn prefix_cache_byte_capacity(&self) -> usize {
        self.prefixes.byte_capacity()
    }

    pub(super) fn prefix_cache_resident_bytes(&self) -> usize {
        self.prefixes.resident_bytes()
    }

    pub(super) fn clear_prefix_cache(&mut self) {
        self.prefixes.clear();
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn embed(&self, token_ids: &[u32], dimensions: usize) -> Result<Vec<f32>> {
        let LoadedExecution::Embedding(model) = &self.execution else {
            return Err(Error::UnsupportedModel("loaded task does not expose embeddings".into()));
        };
        Ok(model.embed(token_ids, dimensions, &self.stream)?)
    }

    pub fn score(&self, token_ids: &[u32]) -> Result<f32> {
        let LoadedExecution::SequenceScoring(model) = &self.execution else {
            return Err(Error::UnsupportedModel(
                "loaded task does not expose sequence scores".into(),
            ));
        };
        Ok(model.score(token_ids, &self.stream)?)
    }

    #[must_use]
    pub(super) fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.execution.decoder().map_or((0, 0, 0, 0), DecoderModel::fusion_summary)
    }

    pub(super) fn expert_fusion_summary(&self) -> String {
        self.execution.decoder().map_or_else(
            |_| "expert fusion is not applicable to this task".into(),
            DecoderModel::expert_fusion_summary,
        )
    }

    pub(super) fn prefill_chunk_len(&self, _position: usize, remaining: usize) -> usize {
        remaining.min(self.info.prefill_step)
    }
}
