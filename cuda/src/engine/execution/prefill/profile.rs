use crate::{
    Result,
    engine::{CudaEngine, model::ModelExecution},
};

impl CudaEngine {
    /// Scheduling capability of the loaded generation runner.
    pub fn prefill_schedule(&self, model_id: &str) -> Result<crate::CudaPrefillSchedule> {
        let loaded = self.model(model_id)?;
        let runner = loaded.prefill_runner()?;
        Ok(match &runner.execution {
            ModelExecution::Generation(generation) => generation.prefill_schedule(),
            _ => crate::CudaPrefillSchedule::RoundRobin,
        })
    }

    /// Exact full and interleaved prefill shapes to calibrate before serving.
    pub fn prefill_profile_shapes(
        &self,
        model_id: &str,
        max_batch_tokens: usize,
    ) -> Result<Vec<usize>> {
        let budget = max_batch_tokens.saturating_sub(1);
        let loaded = self.model(model_id)?;
        let runner = loaded.prefill_runner()?;
        Ok(match &runner.execution {
            ModelExecution::Generation(generation) => {
                let interleaved = generation.interleaved_prefill_budget(budget);
                let mut shapes = Vec::new();
                if interleaved > 0 && interleaved < budget {
                    let full = generation.prefill_chunk_len(max_batch_tokens);
                    if full > 0 {
                        shapes.push(full);
                    }
                    if full != interleaved {
                        shapes.push(interleaved);
                    }
                }
                shapes
            },
            ModelExecution::Embedding(_) | ModelExecution::SequenceScoring(_) => Vec::new(),
        })
    }

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
