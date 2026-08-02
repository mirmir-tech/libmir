use crate::{
    Result,
    engine::{CudaEngine, model::ModelExecution},
};

impl CudaEngine {
    pub fn supports_paged_prefix_reuse(&self, model_id: &str) -> Result<bool> {
        Ok(self.paged_prefix_replay_tokens(model_id)?.is_some())
    }

    pub fn paged_prefix_replay_tokens(&self, model_id: &str) -> Result<Option<usize>> {
        let loaded = self.model(model_id)?;
        let runner = loaded.prefill_runner()?;
        Ok(match &runner.execution {
            ModelExecution::Generation(generation) => generation.prefix_replay_tokens(),
            ModelExecution::Embedding(_) | ModelExecution::SequenceScoring(_) => None,
        })
    }

    pub fn paged_prefix_admission(&self, model_id: &str) -> Result<Option<(usize, usize, usize)>> {
        let loaded = self.model(model_id)?;
        let runner = loaded.prefill_runner()?;
        Ok(match &runner.execution {
            ModelExecution::Generation(generation) => generation.prefix_admission(),
            ModelExecution::Embedding(_) | ModelExecution::SequenceScoring(_) => None,
        })
    }
}
