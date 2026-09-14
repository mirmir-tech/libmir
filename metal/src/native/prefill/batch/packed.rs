use std::time::Instant;

use super::{Batch, Sequence};
use crate::{
    MetalProgressEvent,
    native::{error::Result, model::LoadedModel},
};

pub(super) enum PackedStep {
    Advanced(usize, Vec<(usize, MetalProgressEvent)>),
    #[cfg(test)]
    InsufficientBudget,
    Unavailable,
}

#[cfg(test)]
pub(super) enum RemainderPolicy {
    Consume,
    PreserveCohort,
}
const MAX_PACKED_PREFILL_TOKEN_PAIRS: usize = 32 * 1_024 * 1_024;

impl Batch {
    pub(super) fn execute_packed(
        &mut self,
        loaded: &mut LoadedModel,
        budget: usize,
    ) -> Result<PackedStep> {
        self.workspace_constrained = false;
        let model = loaded.execution.decoder()?;
        if !model.supports_packed_prefill() {
            return Ok(PackedStep::Unavailable);
        }
        let Some(position) = self
            .sequences
            .iter()
            .find(|sequence| sequence.prefill_count(loaded, 1).is_some())
            .map(|sequence| sequence.position)
        else {
            return Ok(PackedStep::Unavailable);
        };
        #[cfg(test)]
        let row_limit = match self.remainder_policy {
            RemainderPolicy::Consume => budget,
            RemainderPolicy::PreserveCohort => usize::MAX,
        };
        #[cfg(not(test))]
        let row_limit = budget;
        let candidates = self
            .sequences
            .iter()
            .enumerate()
            .filter(|(_, sequence)| {
                sequence.position == position && sequence.prefill_count(loaded, 1).is_some()
            })
            .map(|(row, _)| row)
            .take(row_limit)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Ok(PackedStep::Unavailable);
        }
        #[cfg(test)]
        if candidates.len() > budget {
            // A remainder smaller than the cohort must not move only some rows
            // forward: subsequent steps would lose their common position.
            return Ok(PackedStep::InsufficientBudget);
        }
        let row_budget = budget / candidates.len();
        let count = candidates
            .iter()
            .filter_map(|row| self.sequences[*row].prefill_count(loaded, row_budget))
            .min()
            .unwrap_or(0);
        if count == 0 {
            return Ok(PackedStep::Unavailable);
        }
        let work_fits = packed_prefill_work_fits(candidates.len(), position, count);
        let memory_fits =
            work_fits && loaded.packed_prefill_fits(candidates.len(), position, count)?;
        if !work_fits || !memory_fits {
            self.workspace_constrained = work_fits && !memory_fits;
            tracing::info!(
                batch = candidates.len(),
                position,
                sequence = count,
                memory_constrained = self.workspace_constrained,
                "using sequential Metal prefill to preserve workspace headroom"
            );
            return Ok(PackedStep::Unavailable);
        }
        let mut selected = vec![false; self.sequences.len()];
        for row in &candidates {
            selected[*row] = true;
        }
        let mut sequences = self
            .sequences
            .iter_mut()
            .enumerate()
            .filter_map(|(row, sequence)| selected[row].then_some(sequence))
            .collect::<Vec<&mut Sequence>>();
        let advancing = Instant::now();
        Sequence::advance_packed(loaded, &mut sequences, count)?;
        let elapsed = advancing.elapsed();
        for sequence in sequences {
            sequence.timing.record(elapsed);
        }
        let events = candidates
            .iter()
            .map(|row| {
                let sequence = &self.sequences[*row];
                (
                    *row,
                    MetalProgressEvent::prefill_tokens(
                        sequence.position,
                        sequence.request.prompt_tokens.len(),
                    ),
                )
            })
            .collect();
        self.cursor = (candidates[candidates.len() - 1] + 1) % self.sequences.len();
        Ok(PackedStep::Advanced(count * candidates.len(), events))
    }
}

fn packed_prefill_work_fits(batch: usize, position: usize, sequence: usize) -> bool {
    batch.saturating_mul(sequence).saturating_mul(position.saturating_add(sequence))
        <= MAX_PACKED_PREFILL_TOKEN_PAIRS
}

#[cfg(test)]
mod tests {
    use super::packed_prefill_work_fits;

    #[test]
    fn bounds_packed_prefill_by_attention_work() {
        assert!(packed_prefill_work_fits(10, 8_192, 204));
        assert!(!packed_prefill_work_fits(10, 16_384, 204));
        assert!(packed_prefill_work_fits(2, 16_384, 512));
    }
}
