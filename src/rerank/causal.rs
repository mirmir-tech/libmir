use foundation::conversation::{Conversation, Message};
use models::execution::CausalScoringTask;
use runtime::backend::SamplingLogits;

use super::RerankRequest;
use crate::{Error, Model, Result};

pub(super) fn score(
    model: &Model,
    request: &RerankRequest,
    task: &CausalScoringTask,
    vocab_size: usize,
    max_length: usize,
) -> Result<(Vec<f32>, usize)> {
    let descriptor = model.descriptor();
    // Prepare every candidate before starting GPU work. Never truncate the rendered
    // assistant suffix: its final token defines the yes/no scoring position.
    let prompts = request
        .documents
        .iter()
        .map(|document| {
            let conversation = conversation(task, &request.query, document);
            let prompt = descriptor.template().render(&conversation)?;
            let tokens = descriptor
                .tokenizer()
                .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?
                .token_ids;
            if tokens.is_empty() {
                return Err(Error::EmptyPrompt);
            }
            if tokens.len() > max_length {
                return Err(Error::Context {
                    requested: tokens.len(),
                    context: max_length,
                    prompt: tokens.len(),
                    max_tokens: 0,
                });
            }
            Ok(tokens)
        })
        .collect::<Result<Vec<_>>>()?;
    let prompt_tokens = prompts.iter().map(Vec::len).sum();
    let mut scores = Vec::with_capacity(prompts.len());
    for tokens in &prompts {
        let output = model.session().prefill(tokens, SamplingLogits::Full, &mut |_| {})?;
        let logits = output.logits.ok_or_else(|| invalid("causal reranker returned no logits"))?;
        if logits.values.len() != vocab_size {
            return Err(invalid("causal reranker must return one complete vocabulary row"));
        }
        scores.push(logit_difference(&logits.values, task)?);
    }
    Ok((scores, prompt_tokens))
}

fn conversation(task: &CausalScoringTask, query: &str, document: &str) -> Conversation {
    let mut messages = Vec::new();
    if let Some(instruction) = &task.instruction {
        messages.push(message("system", instruction));
    }
    messages.push(message("query", query));
    messages.push(message("document", document));
    Conversation { messages, ..Conversation::default() }
}

fn message(role: &str, content: &str) -> Message {
    Message {
        role: role.into(),
        content: content.into(),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn logit_difference(logits: &[f32], task: &CausalScoringTask) -> Result<f32> {
    let yes = logits
        .get(task.true_token_id as usize)
        .ok_or_else(|| invalid("true token is outside returned logits"))?;
    let no = logits
        .get(task.false_token_id as usize)
        .ok_or_else(|| invalid("false token is outside returned logits"))?;
    let score = yes - no;
    if !yes.is_finite() || !no.is_finite() || !score.is_finite() {
        return Err(invalid("causal reranker returned non-finite logits"));
    }
    Ok(score)
}

fn invalid(message: &str) -> Error {
    models::ModelsError::InvalidConfig(message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_yes_no_odds_and_rejects_invalid_logits() -> Result<()> {
        let task = CausalScoringTask {
            true_token_id: 2,
            false_token_id: 0,
            instruction: None,
        };
        assert!((logit_difference(&[2.0, 100.0, 5.0], &task)? - 3.0).abs() < f32::EPSILON);
        assert!(logit_difference(&[2.0], &task).is_err());
        assert!(logit_difference(&[f32::INFINITY, 0.0, 1.0], &task).is_err());
        assert!(logit_difference(&[1.0, 0.0, f32::NAN], &task).is_err());
        Ok(())
    }

    #[test]
    fn preserves_checkpoint_roles_and_optional_instruction() {
        let mut task = CausalScoringTask {
            true_token_id: 2,
            false_token_id: 0,
            instruction: None,
        };
        assert_eq!(conversation(&task, "q", "d").messages[0].role, "query");
        task.instruction = Some("retrieve".into());
        let conversation = conversation(&task, "q", "d");
        assert_eq!(conversation.messages[0].content, "retrieve");
        assert_eq!(conversation.messages[2].role, "document");
    }
}
