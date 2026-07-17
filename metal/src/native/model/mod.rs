use std::collections::HashMap;

use foundation::model::ModelManifest;
use models::{
    execution::ExecutionPlan,
    layout::{DecoderConfig, ModelLayout, ModelMetadata},
    tokenizer::TokenizerInfo,
};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{
    error::{Error, Result},
    prefix::PrefixCache,
    session::SessionState,
};
use crate::engine::{Array, DecoderModel, MemoryStats, Stream};

mod batch;
mod load;

pub(super) use batch::DecodeInput;

pub(super) const KV_CACHE_STEP: usize = 256;
const PREFILL_STEP: usize = 512;
const HYBRID_LINEAR_PREFILL_STEP: usize = 2_048;

#[derive(Debug)]
pub(super) struct ModelInfo {
    pub manifest: ModelManifest,
    pub layout: ModelLayout,
    pub metadata: ModelMetadata,
    pub decoder: DecoderConfig,
    pub plan: ExecutionPlan,
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
    pub(super) model: DecoderModel,
    pub(super) prefixes: PrefixCache,
    pub(super) sessions: HashMap<Uuid, SessionState>,
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
        let model = &self.model;
        let stream = &self.stream;
        let state = self.sessions.get_mut(&session).ok_or_else(|| Error::Session {
            model: self.info.manifest.id.clone(),
            session,
        })?;
        if state.pending.is_some() {
            return super::step::decode_pending(model, stream, state, token, sampling);
        }
        let logits =
            super::step::forward_token(model, stream, state, token, state.position, false)?;
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

    pub(super) fn release_session(&mut self, session: Uuid) {
        let _removed = self.sessions.remove(&session);
    }

    pub(super) const fn prefix_cache_enabled(&self) -> bool {
        self.prefixes.enabled()
    }

    pub(super) const fn prefix_cache_capacity(&self) -> usize {
        self.prefixes.capacity()
    }

    pub(super) fn clear_prefix_cache(&mut self) {
        self.prefixes.clear();
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    #[must_use]
    pub(super) fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.model.fusion_summary()
    }

    pub(super) fn expert_fusion_summary(&self) -> String {
        self.model.expert_fusion_summary()
    }

    pub(super) fn prefill_chunk_len(&self, _position: usize, remaining: usize) -> usize {
        remaining.min(self.info.prefill_step)
    }
}

fn prefill_step(
    archetype: models::execution::DecoderArchetype,
    configured: Option<usize>,
) -> usize {
    configured
        .filter(|step| *step > 0)
        .unwrap_or_else(|| default_prefill_step(archetype))
}

const fn default_prefill_step(archetype: models::execution::DecoderArchetype) -> usize {
    match archetype {
        models::execution::DecoderArchetype::HybridLinearMoe => HYBRID_LINEAR_PREFILL_STEP,
        models::execution::DecoderArchetype::HybridMoe
        | models::execution::DecoderArchetype::DenseSwiGlu => PREFILL_STEP,
    }
}

#[cfg(test)]
mod tests {
    use models::execution::DecoderArchetype;

    use super::*;

    #[test]
    fn gives_hybrid_linear_moe_a_larger_default_prefill_graph() {
        assert_eq!(default_prefill_step(DecoderArchetype::HybridLinearMoe), 2_048);
        assert_eq!(default_prefill_step(DecoderArchetype::HybridMoe), 512);
        assert_eq!(default_prefill_step(DecoderArchetype::DenseSwiGlu), 512);
    }
}
