mod special;
mod state;

use std::path::Path;

pub use state::{DecisionState, StateEnd};
use tokenizers::Tokenizer;

use super::{AgentConfig, Question, QuestionKind};
use crate::error::{ModelsError, Result};

/// Tokens kept of one rendered option, before the head budget applies.
const OPTION_TOKENS: usize = 48;
/// Minimum head budget left for the instruction before options are cut.
const INSTRUCTION_RESERVE: usize = 16;
/// Tokens every option keeps when the head budget forces a cut.
const MINIMUM_OPTION_TOKENS: usize = 4;
/// Tokens the instruction keeps however long its options are.
const MINIMUM_INSTRUCTION_TOKENS: usize = 8;

/// One question asked of one state:
/// `[CLS] <kind> question: <instruction> [SEP] [MASK] option… [SEP] state
/// [SEP]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRow {
    pub tokens: Vec<u32>,
    /// Position of the `[MASK]` that opens each option, in option order.
    pub markers: Vec<usize>,
    pub kind: QuestionKind,
    /// State tokens that did not fit and were left out.
    pub state_tokens_dropped: usize,
}

/// Tokenizer and special tokens of a Laya checkpoint.
#[derive(Debug, Clone)]
pub struct DecisionTokenizer {
    tokenizer: Tokenizer,
    special: special::SpecialTokens,
}

impl DecisionTokenizer {
    pub fn from_files(tokenizer: impl AsRef<Path>, config: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer)?;
        let special = special::SpecialTokens::resolve(&tokenizer, config.as_ref())?;
        Ok(Self { tokenizer, special })
    }

    /// Builds one row per question, in question order.
    pub fn rows(
        &self,
        state: &DecisionState,
        questions: &[Question],
        config: &AgentConfig,
    ) -> Result<Vec<DecisionRow>> {
        let state_tokens = self.encode(state.as_str())?;
        questions
            .iter()
            .map(|question| self.row(&state_tokens, state.kept(), question, config))
            .collect()
    }

    fn row(
        &self,
        state: &[u32],
        kept: StateEnd,
        question: &Question,
        config: &AgentConfig,
    ) -> Result<DecisionRow> {
        let special = &self.special;
        let mut options = question
            .rendered_options()
            .iter()
            .map(|option| {
                let mut tokens = vec![special.mask];
                tokens.extend(self.encode(&format!(" {option}"))?.into_iter().take(OPTION_TOKENS));
                Ok(tokens)
            })
            .collect::<Result<Vec<_>>>()?;
        let head = config.head_max_len;
        if option_tokens(&options) + INSTRUCTION_RESERVE > head {
            let each = (head.saturating_sub(INSTRUCTION_RESERVE) / options.len().max(1))
                .max(MINIMUM_OPTION_TOKENS);
            for option in &mut options {
                option.truncate(each);
            }
        }
        let instruction_budget =
            head.saturating_sub(option_tokens(&options)).max(MINIMUM_INSTRUCTION_TOKENS);
        let prompt =
            format!("{} question: {}", question.kind().prompt_name(), question.instruction());
        let mut tokens = vec![special.cls];
        tokens.extend(self.encode(&prompt)?.into_iter().take(instruction_budget));
        tokens.push(special.sep);
        let mut markers = Vec::with_capacity(options.len());
        for option in &options {
            markers.push(tokens.len());
            tokens.extend(option);
        }
        tokens.push(special.sep);
        let room = config.max_len.saturating_sub(tokens.len() + 1);
        let kept_state = match kept {
            StateEnd::Start => &state[..room.min(state.len())],
            StateEnd::End => &state[state.len().saturating_sub(room)..],
        };
        tokens.extend(kept_state);
        tokens.push(special.sep);
        tokens.truncate(config.max_len);
        if markers.iter().any(|&marker| marker >= config.max_len) {
            return Err(ModelsError::InvalidConfig(format!(
                "the options of `{}` do not fit in max_len={}; use fewer or shorter options",
                question.instruction(),
                config.max_len
            )));
        }
        Ok(DecisionRow {
            tokens,
            markers,
            kind: question.kind(),
            state_tokens_dropped: state.len() - kept_state.len(),
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let text = text.replace(&self.special.mask_text, " ");
        Ok(self.tokenizer.encode(text, false)?.get_ids().to_vec())
    }
}

fn option_tokens(options: &[Vec<u32>]) -> usize {
    options.iter().map(Vec::len).sum()
}

#[cfg(test)]
mod tests;
