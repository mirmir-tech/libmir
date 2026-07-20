mod discovery;
mod sentence_transformers;

use std::collections::BTreeMap;

pub use discovery::TaskExecutionPlan;

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
}
