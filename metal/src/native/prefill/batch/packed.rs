use super::{Batch, Sequence};
use crate::{
    MetalProgressEvent,
    native::{error::Result, model::LoadedModel},
};

type PackedStep = Option<(usize, Vec<(usize, MetalProgressEvent)>)>;

impl Batch {
    pub(super) fn execute_packed(
        &mut self,
        loaded: &LoadedModel,
        budget: usize,
    ) -> Result<PackedStep> {
        if !loaded.execution.decoder()?.supports_packed_prefill() {
            return Ok(None);
        }
        let candidates = self
            .sequences
            .iter()
            .enumerate()
            .filter(|(_, sequence)| {
                sequence.packed_prefill_eligible() && sequence.prefill_count(loaded, 1).is_some()
            })
            .map(|(row, _)| row)
            .take(budget)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Ok(None);
        }
        let row_budget = budget / candidates.len();
        let count = candidates
            .iter()
            .filter_map(|row| self.sequences[*row].prefill_count(loaded, row_budget))
            .min()
            .unwrap_or(0);
        if count == 0 {
            return Ok(None);
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
        Sequence::advance_packed(loaded, &mut sequences, count)?;
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
        Ok(Some((count * candidates.len(), events)))
    }
}
