use models::chat::ToolCallPrefix;

use super::{GenerationRequest, PreparedGeneration, budget::Allowance, invalid};
use crate::{Model, ReasoningMode, Result};

pub(super) enum Phase {
    /// The model may open one of its reasoning blocks, as (opener, closer)
    /// pairs, or start the constrained output directly.
    Opening {
        blocks: Vec<(u32, u32)>,
        allowance: Allowance,
    },
    Reasoning {
        end: u32,
        allowance: Allowance,
    },
    Grammar,
}

impl Phase {
    pub(super) fn prepare(
        model: &Model,
        request: &GenerationRequest,
        prepared: &PreparedGeneration,
    ) -> Result<(ToolCallPrefix, Self)> {
        if let Some(prefix) = prepared.tool_prefix() {
            if request.reasoning_token_budget.is_some() {
                return Err(invalid("reasoning token budget requires a native reasoning phase"));
            }
            return Ok((prefix.clone(), Self::Grammar));
        }
        let descriptor = model.descriptor();
        let prefix = descriptor
            .template()
            .named_tool_prefix(&request.conversation)?
            .ok_or_else(|| invalid("schema mode requires an explicitly named XML tool"))?;
        let end = descriptor
            .tokenizer()
            .xml_reasoning_end_token(prepared.prompt_text())
            .ok_or_else(|| {
                invalid("schema reasoning requires a native think prefix and atomic closing token")
            })?;
        Ok((
            prefix,
            Self::Reasoning {
                end,
                allowance: Allowance::new(request.reasoning_token_budget),
            },
        ))
    }

    /// The phase before constrained content: inside a reasoning block the
    /// prompt already opened, free to open one when reasoning is enabled, or
    /// straight in the grammar.
    pub(super) fn prepare_content(
        model: &Model,
        request: &GenerationRequest,
        prepared: &PreparedGeneration,
    ) -> Result<Self> {
        let tokenizer = model.descriptor().tokenizer();
        let allowance = Allowance::new(request.reasoning_token_budget);
        if let Some(end) = tokenizer.xml_reasoning_end_token(prepared.prompt_text()) {
            return Ok(Self::Reasoning { end, allowance });
        }
        if request.reasoning != ReasoningMode::Enabled {
            if request.reasoning_token_budget.is_some() {
                return Err(invalid("reasoning token budget requires enabled reasoning"));
            }
            return Ok(Self::Grammar);
        }
        let blocks = tokenizer.reasoning_block_tokens();
        if blocks.is_empty() {
            return Err(invalid("enabled reasoning needs atomic reasoning block tokens"));
        }
        Ok(Self::Opening { blocks, allowance })
    }

    pub(super) fn is_reasoning(&self) -> bool {
        matches!(self, Self::Reasoning { .. })
    }

    pub(super) const fn is_grammar(&self) -> bool {
        matches!(self, Self::Grammar)
    }

    /// Tokens that open a reasoning block, allowed beside the grammar's own.
    pub(super) fn openers(&self) -> impl Iterator<Item = u32> + '_ {
        let blocks = match self {
            Self::Opening { blocks, .. } => blocks.as_slice(),
            Self::Reasoning { .. } | Self::Grammar => &[],
        };
        blocks.iter().map(|(open, _)| *open)
    }

    pub(super) fn end_token(&self) -> Option<u32> {
        match self {
            Self::Reasoning { end, .. } => Some(*end),
            Self::Opening { .. } | Self::Grammar => None,
        }
    }

    pub(super) fn reasoning_bounds(&self) -> Option<(u32, Option<usize>)> {
        match self {
            Self::Reasoning { end, allowance: Allowance::Unlimited } => Some((*end, None)),
            Self::Reasoning {
                end,
                allowance: Allowance::Remaining(left),
            } => Some((*end, Some(left.get()))),
            Self::Reasoning { allowance: Allowance::Exhausted, .. }
            | Self::Opening { .. }
            | Self::Grammar => None,
        }
    }

    pub(super) fn budget_exit(&self) -> Option<u32> {
        match self {
            Self::Reasoning { end, allowance: Allowance::Exhausted } => Some(*end),
            _ => None,
        }
    }

    /// Reasoning tokens, including the transition delimiters, are never fed
    /// to the grammar. From the next token onward its mask is authoritative.
    /// Returns whether the phase took `token`.
    pub(super) fn observe(&mut self, token: u32) -> bool {
        match self {
            Self::Opening { blocks, allowance } => {
                let Some(&(_, end)) = blocks.iter().find(|(open, _)| *open == token) else {
                    *self = Self::Grammar;
                    return false;
                };
                let allowance = std::mem::replace(allowance, Allowance::Unlimited);
                *self = Self::Reasoning { end, allowance };
                true
            },
            Self::Reasoning { end, allowance } => {
                if token == *end {
                    *self = Self::Grammar;
                } else {
                    allowance.consume();
                }
                true
            },
            Self::Grammar => false,
        }
    }
}
