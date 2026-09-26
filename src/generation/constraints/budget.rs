use std::num::NonZeroUsize;

use runtime::backend::{DeviceSampling, SamplingLogits, TokenMask};

use super::{GenerationRequest, ToolConstraints, invalid};
use crate::{ReasoningMode, Result};

pub(super) enum Allowance {
    Unlimited,
    Remaining(NonZeroUsize),
    Exhausted,
}

impl Allowance {
    pub(super) fn new(limit: Option<NonZeroUsize>) -> Self {
        limit.map_or(Self::Unlimited, Self::Remaining)
    }

    pub(super) fn consume(&mut self) {
        if let Self::Remaining(left) = self {
            *self = NonZeroUsize::new(left.get() - 1).map_or(Self::Exhausted, Self::Remaining);
        }
    }
}

pub(super) fn validate(request: &GenerationRequest, total: usize) -> Result<()> {
    let Some(limit) = request.reasoning_token_budget else {
        return Ok(());
    };
    if request.reasoning != ReasoningMode::Enabled
        || request.tool_constraints != ToolConstraints::Schema
    {
        return Err(invalid(
            "a reasoning token budget requires enabled reasoning and schema tools",
        ));
    }
    if limit.get().checked_add(2).is_none_or(|minimum| minimum > total) {
        return Err(invalid("reasoning token budget must reserve the delimiter and tool output"));
    }
    Ok(())
}

pub(super) fn sampling(end: u32, vocab: usize) -> Result<SamplingLogits> {
    let end = end as usize;
    if end >= vocab {
        return Err(invalid("reasoning delimiter is outside the tokenizer vocabulary"));
    }
    let mut words = vec![0; vocab.div_ceil(32)];
    words[end / 32] |= 1 << (end % 32);
    Ok(SamplingLogits::Masked {
        mask: TokenMask::new(words, vocab)?,
        sampling: DeviceSampling::Greedy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowance_never_expands_the_total_completion_limit() {
        let mut request = GenerationRequest {
            reasoning: ReasoningMode::Enabled,
            tool_constraints: ToolConstraints::Schema,
            reasoning_token_budget: NonZeroUsize::new(10),
            ..GenerationRequest::default()
        };
        assert!(validate(&request, 12).is_ok());
        assert!(validate(&request, 11).is_err());
        assert!(validate(&request, 0).is_err());
        request.reasoning_token_budget = NonZeroUsize::new(usize::MAX);
        assert!(validate(&request, usize::MAX).is_err());
        request.reasoning_token_budget = NonZeroUsize::new(10);
        request.reasoning = ReasoningMode::Disabled;
        assert!(validate(&request, 100).is_err());
        request.reasoning = ReasoningMode::Enabled;
        request.tool_constraints = ToolConstraints::None;
        assert!(validate(&request, 100).is_err());
        request.reasoning_token_budget = None;
        assert!(validate(&request, 100).is_ok());
    }
}
