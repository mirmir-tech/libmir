use std::{
    collections::HashSet,
    sync::{Mutex, MutexGuard},
};

use foundation::model::ModelManifest;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    layout::{DecoderConfig, EncoderConfig, ModelLayout, ModelMetadata, VisionConfig},
    weights::{TensorCatalog, TensorReadiness},
};
use uuid::Uuid;

use super::{
    batch::DecodeBuckets,
    runner::{RunnerGuard, RunnerQueue},
    vision::model::LoadedVisionModel,
};
use crate::{Error, Result, backend::CudaTextEmbeddingModel};

mod capacity;
mod generation;
mod load;
mod projection;

pub(super) use generation::{
    GenerationExecution, PooledVisionPrefill, PrefillChunk, SpatialVisionPrefill,
};

pub(super) struct LoadedModel {
    pub manifest: ModelManifest,
    pub layout: ModelLayout,
    pub metadata: ModelMetadata,
    pub decoder: Option<DecoderConfig>,
    pub encoder: Option<EncoderConfig>,
    pub catalog: TensorCatalog,
    pub contract: Option<DecoderExecutionContract>,
    pub task_plan: TaskExecutionPlan,
    pub vision: Option<VisionConfig>,
    pub vision_readiness: Option<TensorReadiness>,
    pub vision_model: Option<LoadedVisionModel>,
    sessions: Mutex<HashSet<Uuid>>,
    runner: RunnerQueue<ModelRunner>,
}

pub(super) struct ModelRunner {
    pub execution: ModelExecution,
    pub batches: Option<DecodeBuckets>,
    pub selected: Option<DeviceToken>,
}

pub(super) enum ModelExecution {
    Generation(Box<dyn GenerationExecution>),
    Embedding(Box<CudaTextEmbeddingModel>),
    SequenceScoring(Box<crate::backend::CudaSequenceScoringModel>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DeviceToken {
    pub session: Uuid,
    pub token: u32,
}

impl LoadedModel {
    pub(super) fn semantic(&self) -> Option<&models::semantic::SemanticModelSpec> {
        self.contract.as_ref().map(|contract| &contract.semantic)
    }

    pub(super) fn sessions(&self) -> Result<MutexGuard<'_, HashSet<Uuid>>> {
        let Ok(sessions) = self.sessions.lock() else {
            return Err(Error::State("session registry lock is poisoned".into()));
        };
        Ok(sessions)
    }

    pub fn clear_sessions(&self) -> Result<()> {
        let mut runner = self.prefill_runner()?;
        self.sessions()?.clear();
        if let ModelExecution::Generation(generation) = &mut runner.execution {
            generation.clear_sessions();
        }
        runner.selected = None;
        drop(runner);
        Ok(())
    }

    pub(super) fn register_session(&self, session: Uuid) -> Result<()> {
        self.sessions()?.insert(session);
        Ok(())
    }

    pub fn release_session(&self, session: Uuid) -> Result<()> {
        let mut runner = self.prefill_runner()?;
        self.sessions()?.remove(&session);
        if let ModelExecution::Generation(generation) = &mut runner.execution {
            generation.release_session(session);
        }
        if runner.selected.is_some_and(|selected| selected.session == session) {
            runner.selected = None;
        }
        drop(runner);
        Ok(())
    }

    pub(super) fn require_session(&self, session: Uuid) -> Result<()> {
        if !self.sessions()?.contains(&session) {
            return Err(Error::State("decode session is not initialized".into()));
        }
        Ok(())
    }

    pub(super) fn decode_runner(&self) -> Result<RunnerGuard<'_, ModelRunner>> {
        self.runner.acquire_decode()
    }

    pub(super) fn prefill_runner(&self) -> Result<RunnerGuard<'_, ModelRunner>> {
        self.runner.acquire_prefill()
    }
}
