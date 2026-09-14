mod cancellation;
mod execution;
mod packed;
mod reservation;
mod retention;
pub(in crate::native) use retention::Reservations;
mod sequence;
#[cfg(test)]
mod tests;

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

pub(in crate::native) use cancellation::RetirePrefill;
use runtime::backend::{PrefillRequest, SamplingLogits};

use self::{packed::PackedStep, sequence::Sequence};
use super::{MetalPrefillCohort, NativePrefill, cohort::restore_prefix, timing::PrefillTiming};
use crate::{
    MetalProgressEvent,
    native::{
        error::{Error, Result},
        model::LoadedModel,
    },
};

/// A shared handle to a prepared prefill group.
///
/// Dropping its last handle before successful finish schedules retirement on
/// the model worker; dropping a clone does not cancel other handles or wait for
/// GPU work.
#[derive(Clone)]
pub struct MetalPrefillBatch {
    model_id: String,
    inner: Arc<Mutex<Option<Batch>>>,
    _cleanup: Arc<cancellation::Cleanup>,
}

pub(in crate::native) struct PrefillStep {
    pub(in crate::native) events: Vec<(usize, MetalProgressEvent)>,
    pub(in crate::native) complete: bool,
}

pub(in crate::native) struct FinishedPrefill {
    pub(in crate::native) request: PrefillRequest,
    pub(in crate::native) native: NativePrefill,
    pub(in crate::native) timing: PrefillTiming,
}

pub(in crate::native) struct Batch {
    sequences: Vec<Sequence>,
    cursor: usize,
    workspace_constrained: bool,
    #[cfg(test)]
    remainder_policy: packed::RemainderPolicy,
}

impl MetalPrefillBatch {
    #[cfg(test)]
    pub(in crate::native) fn prepare(
        loaded: &mut LoadedModel,
        requests: Vec<(PrefillRequest, SamplingLogits)>,
        cohort: Option<&MetalPrefillCohort>,
    ) -> Result<(Self, Vec<(usize, MetalProgressEvent)>)> {
        Self::prepare_at(loaded, requests, cohort, Instant::now(), None)
    }

    pub(in crate::native) fn prepare_at(
        loaded: &mut LoadedModel,
        requests: Vec<(PrefillRequest, SamplingLogits)>,
        cohort: Option<&MetalPrefillCohort>,
        submitted: Instant,
        retire: Option<RetirePrefill>,
    ) -> Result<(Self, Vec<(usize, MetalProgressEvent)>)> {
        loaded.require_execution_ready()?;
        let validating = Instant::now();
        super::validation::validate_requests(requests.iter().map(|(request, _)| request))?;
        let mut leased = cohort
            .map(|cohort| cohort.take(requests.iter().map(|(request, _)| request.session_id)))
            .transpose()?
            .unwrap_or_default()
            .into_iter();
        let validated = validating.elapsed();
        let model_id = loaded.info.manifest.id.clone();
        let mut events = Vec::with_capacity(requests.len());
        let mut sequences = requests
            .into_iter()
            .enumerate()
            .map(|(row, (request, execution_sampling))| {
                let preparing = Instant::now();
                let restored = restore_prefix(loaded, &request, leased.next())?;
                let mut sequence = Sequence::prepare(
                    loaded,
                    request,
                    execution_sampling,
                    restored,
                    PrefillTiming::new(submitted),
                )?;
                sequence.timing.record(validated + preparing.elapsed());
                events.push((
                    row,
                    MetalProgressEvent::prefill_tokens(
                        sequence.position,
                        sequence.request.prompt_tokens.len(),
                    ),
                ));
                Ok(sequence)
            })
            .collect::<Result<Vec<_>>>()?;
        let reserving = Instant::now();
        reservation::fit(&mut sequences, loaded)?;
        if cohort.is_none() && loaded.prefixes.reserve_batch_slots(sequences.len()) {
            crate::engine::clear_memory_cache()?;
        }
        let reserved = reserving.elapsed();
        for sequence in &mut sequences {
            sequence.timing.record(reserved);
        }
        let inner = Arc::new(Mutex::new(Some(Batch {
            sequences,
            cursor: 0,
            workspace_constrained: false,
            #[cfg(test)]
            remainder_policy: packed::RemainderPolicy::Consume,
        })));
        if loaded.stream.config().cache.decode_reservation
            == crate::config::DecodeReservation::GenerationBudget
        {
            loaded.prefill_reservations.register(&inner);
        }
        let cleanup = Arc::new(cancellation::Cleanup { inner: Arc::clone(&inner), retire });
        Ok((Self { model_id, inner, _cleanup: cleanup }, events))
    }

    #[cfg(test)]
    pub(in crate::native) fn execute_step(
        &self,
        loaded: &mut LoadedModel,
        token_budget: usize,
    ) -> Result<PrefillStep> {
        self.execute_step_until(loaded, token_budget, &mut || false)
    }

    pub(in crate::native) fn execute_step_until(
        &self,
        loaded: &mut LoadedModel,
        token_budget: usize,
        should_yield: &mut dyn FnMut() -> bool,
    ) -> Result<PrefillStep> {
        loaded.require_execution_ready()?;
        let mut guard = self.inner.lock()?;
        let batch = guard
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill batch was finished".into()))?;
        let result = batch.execute_step(loaded, token_budget.max(1), should_yield);
        match result {
            Ok(step) => {
                drop(guard);
                Ok(step)
            },
            Err(error) => {
                let batch = guard
                    .take()
                    .ok_or_else(|| Error::InvalidPrefillBatch("missing failed batch".into()))?;
                drop(guard);
                batch.fail(loaded, error)
            },
        }
    }

    pub(in crate::native) fn finish(&self) -> Result<Vec<FinishedPrefill>> {
        let mut guard = self.inner.lock()?;
        if let Some(batch) = guard.as_ref()
            && batch.sequences.iter().any(|sequence| sequence.output.is_none())
        {
            return Err(Error::InvalidPrefillBatch("prefill batch is incomplete".into()));
        }
        let batch = guard
            .take()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill batch was finished".into()))?;
        drop(guard);
        batch.finish()
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[cfg(test)]
    pub(in crate::native) fn preserve_cohort_for_benchmark(&self) -> Result<()> {
        let mut guard = self.inner.lock()?;
        let batch = guard
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("missing batch".into()))?;
        batch.remainder_policy = packed::RemainderPolicy::PreserveCohort;
        drop(guard);
        Ok(())
    }
}
