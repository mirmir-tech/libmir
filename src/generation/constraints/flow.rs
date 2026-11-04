use llguidance::toktrie::SimpleVob;
use runtime::backend::{DeviceSampling, SamplingLogits, TokenMask};

use super::{OutputConstraint, Target, budget, invalid, validation};
use crate::Result;

pub(in crate::generation) fn sampling(
    constraint: &mut Option<OutputConstraint>,
    policy: SamplingLogits,
) -> Result<SamplingLogits> {
    let Some(constraint) = constraint else {
        return Ok(policy);
    };
    if let Some(end) = constraint.phase.budget_exit() {
        tracing::warn!("reasoning token budget exit activated");
        return budget::sampling(end, constraint.vocab);
    }
    if constraint.phase.is_reasoning() {
        return Ok(policy);
    }
    let sampling = match policy {
        SamplingLogits::None => DeviceSampling::Greedy,
        SamplingLogits::SampleTopK { k, temperature, draw, .. } => {
            DeviceSampling::Random { top_k: k, top_p: 1.0, temperature, draw }
        },
        SamplingLogits::Sample { top_k, top_p, temperature, draw, .. } => {
            DeviceSampling::Random { top_k, top_p, temperature, draw }
        },
        _ => return Err(invalid("schema mode requires device sampling without history")),
    };
    let bits = constraint.matcher.compute_mask().map_err(|error| invalid(error.to_string()))?;
    let mut words = mask_words(&bits, constraint.vocab)?;
    // Before constrained content the model may still open a reasoning block.
    for opener in constraint.phase.openers() {
        let opener = opener as usize;
        if opener >= constraint.vocab {
            return Err(invalid("reasoning opener is outside the tokenizer vocabulary"));
        }
        words[opener / 32] |= 1 << (opener % 32);
    }
    let mask = TokenMask::new(words, constraint.vocab)?;
    Ok(SamplingLogits::Masked { mask, sampling })
}

/// llguidance allocates one spare bit past the vocabulary, which adds a whole
/// word when the vocabulary is a multiple of 32 (Nemotron-H's 131 072).
pub(super) fn mask_words(bits: &SimpleVob, vocab: usize) -> Result<Vec<u32>> {
    if bits.len() != vocab {
        return Err(invalid("token mask vocabulary mismatch"));
    }
    Ok(bits.as_slice()[..vocab.div_ceil(32)].to_vec())
}

/// A required schema tool cannot finish on the reasoning channel. Use the
/// native delimiter when the model requests end-of-turn there, then generate
/// and validate a fresh tool call. Draft calls inside reasoning stay private.
/// This uses the already-selected token; it adds no accelerator readback.
pub(in crate::generation) fn transition_stop(
    constraint: Option<&OutputConstraint>,
    token: u32,
    stop_tokens: &[u32],
) -> u32 {
    if let Some(end) = constraint.and_then(|c| c.phase.end_token())
        && stop_tokens.contains(&token)
    {
        tracing::warn!(
            stop_token = token,
            "required tool reasoning stop changed to native delimiter"
        );
        end
    } else {
        token
    }
}

pub(in crate::generation) fn advance(
    constraint: &mut Option<OutputConstraint>,
    token: u32,
) -> Result<bool> {
    let Some(constraint) = constraint else {
        return Ok(false);
    };
    constraint.consume(token)?;
    if constraint.target == Target::Content {
        // Accepting means EOS is allowed; a numeric root may still continue.
        Ok(constraint.phase.is_grammar() && constraint.matcher.is_stopped())
    } else {
        constraint.complete()
    }
}

/// Neither recovery nor cycle exits may override a budget delimiter or the
/// tool grammar's device mask.
pub(in crate::generation) fn advance_generation(
    constraint: &mut Option<OutputConstraint>,
    token: u32,
    cycle: &mut crate::generation::CycleRecovery,
) -> Result<bool> {
    let complete = advance(constraint, token)?;
    if constraint
        .as_ref()
        .is_some_and(|c| c.phase.is_grammar() || c.phase.budget_exit().is_some())
    {
        cycle.disable();
    }
    Ok(complete)
}

/// Check grammar completion before parsing an XML/JSON fragment. A token-limit
/// exit can leave delimiters inside an unfinished string; parsing that fragment
/// first misreports a generation truncation as malformed complete arguments.
pub(in crate::generation) fn normalize(
    mut constraint: Option<&mut OutputConstraint>,
    payload: &str,
    conversation: &foundation::conversation::Conversation,
) -> Result<String> {
    if let Some(constraint) = constraint.as_mut()
        && constraint.target == Target::Tool
        && !constraint.complete()?
    {
        return Err(crate::Error::InvalidToolCall(
            "schema-constrained tool generation ended before completion".into(),
        ));
    }
    let calls = crate::generation::tools::normalize(payload, conversation)?;
    if let Some(constraint) = constraint
        && constraint.target == Target::Tool
    {
        validation::validate(&constraint.validator, &calls)?;
    }
    Ok(calls)
}

/// Checks constrained content: complete under the grammar, then valid JSON of
/// the response format's schema.
pub(in crate::generation) fn validate_content(
    constraint: Option<&mut OutputConstraint>,
    content: &str,
) -> Result<()> {
    let Some(constraint) = constraint.filter(|constraint| constraint.target == Target::Content)
    else {
        return Ok(());
    };
    if !constraint.complete()? {
        return Err(crate::Error::InvalidResponseFormat(
            "JSON content ended before its schema was complete".into(),
        ));
    }
    validation::validate_content(&constraint.validator, content)
}
