use models::execution::{ModelTask, TaskExecutionPlan};

use crate::{Error, Model, Result};

#[derive(Debug, Clone)]
/// Batch of text inputs to encode with an embedding checkpoint.
pub struct EmbeddingRequest {
    /// Texts encoded independently in response order.
    pub inputs: Vec<String>,
    /// Optional prefix dimension retained before output normalization.
    pub dimensions: Option<usize>,
    /// Optional prompt preset declared by the checkpoint.
    pub prompt_name: Option<String>,
}

#[derive(Debug, Clone)]
/// Device-computed dense vectors and tokenizer usage.
pub struct EmbeddingOutput {
    /// One normalized vector for every request input.
    pub embeddings: Vec<Vec<f32>>,
    /// Total number of input tokens processed by the backend.
    pub prompt_tokens: usize,
}

impl Model {
    /// Encodes a text batch according to the checkpoint's pooling contract.
    pub fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingOutput> {
        let EmbeddingRequest {
            inputs: requested_inputs,
            dimensions,
            prompt_name,
        } = request;
        let descriptor = self.descriptor();
        let TaskExecutionPlan::Embedding { task, .. } = descriptor.task_plan() else {
            return Err(Error::TaskMismatch {
                requested: "embedding",
                actual: task_name(&descriptor.task()),
            });
        };
        if requested_inputs.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let dimensions = dimensions.unwrap_or(task.native_dimensions);
        if dimensions == 0 || dimensions > task.native_dimensions {
            return Err(models::ModelsError::InvalidConfig(format!(
                "embedding dimensions must be between 1 and {}",
                task.native_dimensions
            ))
            .into());
        }
        let prompt = prompt_name
            .as_deref()
            .or(task.default_prompt.as_deref())
            .map(|name| {
                task.prompts.get(name).cloned().ok_or_else(|| {
                    models::ModelsError::InvalidConfig(format!(
                        "embedding prompt {name:?} is not declared by the checkpoint"
                    ))
                })
            })
            .transpose()?;
        let tokenizer = descriptor.tokenizer();
        let limit = tokenizer
            .default_max_length()
            .unwrap_or_else(|| descriptor.metadata().context_len)
            .min(descriptor.metadata().context_len);
        let inputs: Vec<Vec<u32>> = requested_inputs
            .iter()
            .map(|input| {
                let text = prompt
                    .as_ref()
                    .map_or_else(|| input.clone(), |prompt| format!("{prompt}{input}"));
                Ok(tokenizer.encode_with_limit(&text, limit)?.token_ids)
            })
            .collect::<Result<_>>()?;
        let prompt_tokens = inputs.iter().map(Vec::len).sum();
        Ok(EmbeddingOutput {
            embeddings: self.engine().embed_tokens(self.handle(), &inputs, dimensions)?,
            prompt_tokens,
        })
    }
}

fn task_name(task: &ModelTask) -> &'static str {
    match task {
        ModelTask::Generation => "generation",
        ModelTask::Embedding(_) => "embedding",
        ModelTask::SequenceScoring(_) => "sequence scoring",
    }
}
