//! Bounds of speculative decode steps.

use runtime::{backend::SamplingLogits, speculation::DraftBounds};

use crate::Result;

/// How far the next decode may run ahead of the host, when it takes the
/// backend's tokens as they are: no cycle recovery or host-side sampling.
/// Acceptance stops at a stop token, in a reasoning phase at its delimiter
/// and within its token budget, and under the tool grammar at the reported
/// `drafts` the grammar allows, each verified within its mask.
pub(super) fn draft_bounds(
    constraint: Option<&mut super::constraints::OutputConstraint>,
    cycle: &super::CycleRecovery,
    sampling: &SamplingLogits,
    stop_tokens: &[u32],
    drafts: &[u32],
    remaining_tokens: usize,
) -> Result<Option<DraftBounds>> {
    let completion_limit = remaining_tokens.saturating_sub(1);
    let stop_tokens = stop_tokens.to_vec();
    if let SamplingLogits::Masked { .. } = sampling {
        let masked = match constraint {
            Some(constraint) => constraint.draft_masks(drafts)?,
            None => None,
        };
        return Ok(masked.map(|masked| DraftBounds {
            max_accepted: Some(completion_limit),
            stop_tokens,
            masked,
        }));
    }
    let device = matches!(
        sampling,
        SamplingLogits::None | SamplingLogits::Sample { .. } | SamplingLogits::SampleTopK { .. }
    );
    if !device || !cycle.takes_backend_tokens() {
        return Ok(None);
    }
    let mut stop_tokens = stop_tokens;
    let max_accepted = match constraint {
        None => None,
        Some(constraint) => {
            let Some((end, left)) = constraint.reasoning_bounds() else {
                return Ok(None);
            };
            stop_tokens.push(end);
            // Each kept draft and the step's token spend the budget.
            left.map(|left| left - 1)
        },
    };
    Ok(Some(DraftBounds {
        max_accepted: Some(
            max_accepted.map_or(completion_limit, |limit| limit.min(completion_limit)),
        ),
        stop_tokens,
        masked: Vec::new(),
    }))
}

#[cfg(test)]
mod tests;
