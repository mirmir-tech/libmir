use std::cmp::Ordering;

use models::execution::{ModelTask, TaskExecutionPlan};

use crate::{Error, Model, Result};

#[derive(Debug, Clone)]
/// Query and candidate documents evaluated by a sequence-scoring checkpoint.
pub struct RerankRequest {
    /// Query placed in the first tokenizer sequence.
    pub query: String,
    /// Candidate documents placed in the second tokenizer sequence.
    pub documents: Vec<String>,
    /// Optional pair-token limit, capped by the checkpoint context window.
    pub max_length: Option<usize>,
    /// Returns classifier logits instead of applying the logistic transform.
    pub raw_scores: bool,
}

#[derive(Debug, Clone)]
/// One candidate and its relevance score.
pub struct RerankResult {
    /// Original candidate position before relevance sorting.
    pub index: usize,
    /// Relevance score or raw classifier logit.
    pub score: f32,
    /// Candidate text copied from the request.
    pub document: String,
}

#[derive(Debug, Clone)]
/// Relevance-sorted candidates and tokenizer usage.
pub struct RerankOutput {
    /// Candidates sorted from most to least relevant.
    pub results: Vec<RerankResult>,
    /// Total number of pair tokens processed by the backend.
    pub prompt_tokens: usize,
}

impl Model {
    /// Scores and sorts candidate documents according to the checkpoint
    /// contract.
    pub fn rerank(&self, request: RerankRequest) -> Result<RerankOutput> {
        let descriptor = self.descriptor();
        if !matches!(descriptor.task_plan(), TaskExecutionPlan::SequenceScoring { .. }) {
            return Err(Error::TaskMismatch {
                requested: "sequence scoring",
                actual: task_name(&descriptor.task()),
            });
        }
        if request.query.is_empty() || request.documents.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let tokenizer = descriptor.tokenizer();
        let context = descriptor.metadata().context_len;
        let max_length = request
            .max_length
            .or_else(|| tokenizer.default_max_length())
            .unwrap_or(context)
            .min(context);
        if max_length == 0 {
            return Err(models::ModelsError::InvalidConfig(
                "reranking max_length must be positive".into(),
            )
            .into());
        }
        let pairs: Vec<Vec<u32>> = request
            .documents
            .iter()
            .map(|document| {
                Ok(tokenizer.encode_pair(&request.query, document, max_length)?.token_ids)
            })
            .collect::<Result<_>>()?;
        let prompt_tokens = pairs.iter().map(Vec::len).sum();
        let scores = self.engine().score_tokens(self.handle(), &pairs)?;
        let mut results: Vec<_> = request
            .documents
            .into_iter()
            .zip(scores)
            .enumerate()
            .map(|(index, (document, score))| RerankResult {
                index,
                score: if request.raw_scores {
                    score
                } else {
                    sigmoid(score)
                },
                document,
            })
            .collect();
        results
            .sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap_or(Ordering::Equal));
        Ok(RerankOutput { results, prompt_tokens })
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn task_name(task: &ModelTask) -> &'static str {
    match task {
        ModelTask::Generation => "generation",
        ModelTask::Embedding(_) => "embedding",
        ModelTask::SequenceScoring(_) => "sequence scoring",
    }
}

#[cfg(test)]
mod tests {
    use super::sigmoid;

    #[test]
    fn sigmoid_is_stable_for_large_logits() {
        assert!(sigmoid(1_000.0) > 1.0 - f32::EPSILON);
        assert!(sigmoid(-1_000.0) < f32::EPSILON);
        assert!((sigmoid(0.0) - 0.5).abs() < f32::EPSILON);
    }
}
