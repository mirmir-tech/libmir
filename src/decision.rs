//! Typed decisions with Laya checkpoints.
//!
//! A [`DecisionModel`] answers typed questions about one state in a single
//! encoder pass per question: pick an option, place the state on a scale, or
//! decide whether a statement holds. Enable the `cpu` feature for the native
//! CPU backend, `metal` for Apple GPUs, and `cuda` for NVIDIA GPUs.

use std::path::Path;

pub use models::decision::{
    Answer, ChoiceOption, DecisionCheckpoint, DecisionRow, DecisionSchema, DecisionState, Question,
    QuestionKind, SchemaField, StateEnd, Verdict, Verdicts,
};
use serde_json::{Map, Value};

use crate::Result;

/// Hardware that evaluates a decision checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionBackend {
    /// Native multithreaded CPU execution through `mircup`.
    #[cfg(feature = "cpu")]
    Cpu,
    /// Apple GPU execution through `mirtal`.
    #[cfg(feature = "metal")]
    Metal,
    /// NVIDIA GPU execution through `mircuda`.
    #[cfg(feature = "cuda")]
    Cuda,
}

enum Engine {
    #[cfg(feature = "cpu")]
    Cpu(Box<cpu::CpuDecisionModel>),
    #[cfg(feature = "metal")]
    Metal(Box<metal::engine::MetalDecisionModel>),
    #[cfg(feature = "cuda")]
    Cuda(Box<cuda::CudaDecisionModel>),
}

/// A loaded Laya checkpoint ready to answer questions.
pub struct DecisionModel {
    checkpoint: DecisionCheckpoint,
    engine: Engine,
}

impl std::fmt::Debug for DecisionModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecisionModel")
            .field("root", &self.checkpoint.root)
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl DecisionModel {
    /// Loads the checkpoint directory at `root` onto `backend`.
    pub fn load(root: impl AsRef<Path>, backend: DecisionBackend) -> Result<Self> {
        let checkpoint = DecisionCheckpoint::inspect(root)?;
        let engine = match backend {
            #[cfg(feature = "cpu")]
            DecisionBackend::Cpu => {
                Engine::Cpu(Box::new(cpu::CpuDecisionModel::load(&checkpoint)?))
            },
            #[cfg(feature = "metal")]
            DecisionBackend::Metal => {
                Engine::Metal(Box::new(metal::engine::MetalDecisionModel::load(&checkpoint)?))
            },
            #[cfg(feature = "cuda")]
            DecisionBackend::Cuda => {
                Engine::Cuda(Box::new(cuda::CudaDecisionModel::load(&checkpoint)?))
            },
        };
        Ok(Self { checkpoint, engine })
    }

    /// Hardware this model runs on.
    #[must_use]
    pub const fn backend(&self) -> DecisionBackend {
        match self.engine {
            #[cfg(feature = "cpu")]
            Engine::Cpu(_) => DecisionBackend::Cpu,
            #[cfg(feature = "metal")]
            Engine::Metal(_) => DecisionBackend::Metal,
            #[cfg(feature = "cuda")]
            Engine::Cuda(_) => DecisionBackend::Cuda,
        }
    }

    /// Configuration, tokenizer, and tensor plan of the loaded checkpoint.
    #[must_use]
    pub const fn checkpoint(&self) -> &DecisionCheckpoint {
        &self.checkpoint
    }

    /// Answers every question about `state`, in question order.
    pub fn decide(&self, state: &DecisionState, questions: &[Question]) -> Result<Vec<Answer>> {
        let rows = self.rows(state, questions)?;
        let logits = self.logits(&rows)?;
        questions
            .iter()
            .zip(&logits)
            .map(|(question, logits)| Ok(Answer::decode(question, logits, &self.checkpoint.agent)?))
            .collect()
    }

    /// Answers a flat JSON schema and returns the decided values.
    pub fn decide_schema(
        &self,
        state: &DecisionState,
        schema: &DecisionSchema,
    ) -> Result<Map<String, Value>> {
        Ok(schema.project(&self.decide(state, &schema.questions())?)?)
    }

    /// Token rows the checkpoint reads for each question.
    pub fn rows(&self, state: &DecisionState, questions: &[Question]) -> Result<Vec<DecisionRow>> {
        Ok(self.checkpoint.tokenizer.rows(state, questions, &self.checkpoint.agent)?)
    }

    /// Raw option logits of every row, one per option marker.
    pub fn logits(&self, rows: &[DecisionRow]) -> Result<Vec<Vec<f32>>> {
        match &self.engine {
            #[cfg(feature = "cpu")]
            Engine::Cpu(model) => Ok(model.logits(rows)?),
            #[cfg(feature = "metal")]
            Engine::Metal(model) => Ok(model.logits(rows)?),
            #[cfg(feature = "cuda")]
            Engine::Cuda(model) => Ok(model.logits(rows)?),
        }
    }
}
