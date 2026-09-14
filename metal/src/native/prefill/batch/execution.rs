use super::{
    Batch, Error, FinishedPrefill, Instant, LoadedModel, MetalProgressEvent, PackedStep,
    PrefillStep, Result, Sequence,
};

impl Batch {
    pub(super) fn fail(self, loaded: &mut LoadedModel, error: Error) -> Result<PrefillStep> {
        let sessions = self
            .sequences
            .iter()
            .map(|sequence| sequence.request.session_id)
            .collect::<Vec<_>>();
        let states = self.sequences.into_iter().filter_map(Sequence::into_state);
        loaded.fail_execution(sessions, states, error)
    }

    pub(super) fn execute_step(
        &mut self,
        loaded: &mut LoadedModel,
        mut budget: usize,
        should_yield: &mut dyn FnMut() -> bool,
    ) -> Result<PrefillStep> {
        #[cfg(test)]
        let initial_budget = budget;
        let mut events = Vec::new();
        while budget > 0 && self.sequences.iter().any(|sequence| sequence.output.is_none()) {
            // Yield only between evaluated graphs. The scheduler retires the
            // cancelled rows after receiving this step's original row indices.
            if should_yield() {
                break;
            }
            if loaded.stream.config().cache.decode_reservation
                == crate::config::DecodeReservation::GenerationBudget
            {
                super::reservation::fit(&mut self.sequences, loaded)?;
            }
            match self.execute_packed(loaded, budget)? {
                PackedStep::Advanced(used, packed_events) => {
                    budget -= used;
                    events.extend(packed_events);
                    continue;
                },
                #[cfg(test)]
                PackedStep::InsufficientBudget if budget < initial_budget => break,
                // If the entire configured budget is smaller than a cohort,
                // scalar progress still guarantees that the request can finish.
                #[cfg(test)]
                PackedStep::InsufficientBudget => {},
                PackedStep::Unavailable => {},
            }
            let row = self.cursor % self.sequences.len();
            self.cursor = (self.cursor + 1) % self.sequences.len();
            let sequence = &mut self.sequences[row];
            if sequence.output.is_some() {
                continue;
            }
            let scalar_budget =
                LoadedModel::pressure_bounded_prefill_budget(budget, self.workspace_constrained)?;
            let advancing = Instant::now();
            let used = sequence.advance(loaded, scalar_budget)?;
            sequence.timing.record(advancing.elapsed());
            budget -= used;
            events.push((
                row,
                MetalProgressEvent::prefill_tokens(
                    sequence.position,
                    sequence.request.prompt_tokens.len(),
                ),
            ));
        }
        Ok(PrefillStep {
            events,
            complete: self.sequences.iter().all(|sequence| sequence.output.is_some()),
        })
    }

    pub(super) fn finish(self) -> Result<Vec<FinishedPrefill>> {
        self.sequences
            .into_iter()
            .map(|sequence| {
                let native = sequence.output.ok_or_else(|| {
                    Error::InvalidPrefillBatch("prefill batch is incomplete".into())
                })?;
                Ok(FinishedPrefill {
                    request: sequence.request,
                    native,
                    timing: sequence.timing,
                })
            })
            .collect()
    }
}
