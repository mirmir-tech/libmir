mod discovery;
mod logit_score;
mod sentence_transformers;

use std::{collections::BTreeMap, fmt};

pub use discovery::TaskExecutionPlan;
pub use logit_score::CausalScoringTask;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    Generation,
    Embedding,
    SequenceScoring,
    CausalScoring,
}

impl TaskKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generation => "generation",
            Self::Embedding => "embedding",
            Self::SequenceScoring => "sequence scoring",
            Self::CausalScoring => "reranking",
        }
    }
}

impl fmt::Display for TaskKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMode {
    Cls,
    LastToken,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingTask {
    pub pooling: PoolingMode,
    pub normalize: bool,
    pub native_dimensions: usize,
    pub include_prompt: bool,
    pub prompts: BTreeMap<String, String>,
    pub default_prompt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceScoringTask {
    pub labels: usize,
    pub pooling: PoolingMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelTask {
    Generation,
    Embedding(EmbeddingTask),
    SequenceScoring(SequenceScoringTask),
    CausalScoring(CausalScoringTask),
}

impl ModelTask {
    #[must_use]
    pub const fn kind(&self) -> TaskKind {
        match self {
            Self::Generation => TaskKind::Generation,
            Self::Embedding(_) => TaskKind::Embedding,
            Self::SequenceScoring(_) => TaskKind::SequenceScoring,
            Self::CausalScoring(_) => TaskKind::CausalScoring,
        }
    }
}
