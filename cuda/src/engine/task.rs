use super::{CudaEngine, model};
use crate::{Error, Result};

impl CudaEngine {
    pub fn embed_tokens(
        &self,
        model: &::runtime::backend::ModelHandle,
        inputs: &[Vec<u32>],
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let loaded = self.model(&model.id)?;
        if !matches!(loaded.task_plan, models::execution::TaskExecutionPlan::Embedding { .. }) {
            return Err(Error::State("loaded CUDA task is not an embedding model".into()));
        }
        let mut runner = loaded.prefill_runner()?;
        let model::ModelExecution::Embedding(task) = &mut runner.execution else {
            return Err(Error::State("loaded CUDA task does not expose embeddings".into()));
        };
        let result = inputs.iter().map(|tokens| task.embed(tokens, dimensions)).collect();
        drop(runner);
        result
    }

    pub fn score_tokens(
        &self,
        model: &::runtime::backend::ModelHandle,
        inputs: &[Vec<u32>],
    ) -> Result<Vec<f32>> {
        let loaded = self.model(&model.id)?;
        if !matches!(loaded.task_plan, models::execution::TaskExecutionPlan::SequenceScoring { .. })
        {
            return Err(Error::State("loaded CUDA task is not a sequence-scoring model".into()));
        }
        let mut runner = loaded.prefill_runner()?;
        let model::ModelExecution::SequenceScoring(task) = &mut runner.execution else {
            return Err(Error::State("loaded CUDA task does not expose sequence scores".into()));
        };
        let result = inputs.iter().map(|tokens| task.score(tokens)).collect();
        drop(runner);
        result
    }
}
