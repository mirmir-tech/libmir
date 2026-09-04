mod packed;
mod reservation;
mod sequence;

use std::sync::{Arc, Mutex};

use runtime::backend::{PrefillRequest, SamplingLogits};

use self::sequence::Sequence;
use super::{MetalPrefillCohort, NativePrefill, cohort::restore_prefix, required_prefill_pages};
use crate::{
    MetalProgressEvent,
    native::{
        error::{Error, Result},
        model::LoadedModel,
    },
};

#[derive(Clone)]
pub struct MetalPrefillBatch {
    model_id: String,
    inner: Arc<Mutex<Option<Batch>>>,
}

pub(in crate::native) struct PrefillStep {
    pub(in crate::native) events: Vec<(usize, MetalProgressEvent)>,
    pub(in crate::native) complete: bool,
}

pub(in crate::native) struct FinishedPrefill {
    pub(in crate::native) request: PrefillRequest,
    pub(in crate::native) native: NativePrefill,
    pub(in crate::native) started: std::time::Instant,
}

struct Batch {
    sequences: Vec<Sequence>,
    cursor: usize,
    workspace_constrained: bool,
}

impl MetalPrefillBatch {
    pub(in crate::native) fn prepare(
        loaded: &mut LoadedModel,
        requests: Vec<(PrefillRequest, SamplingLogits)>,
        cohort: Option<&MetalPrefillCohort>,
    ) -> Result<(Self, Vec<(usize, MetalProgressEvent)>)> {
        if requests.is_empty() {
            return Err(Error::InvalidPrefillBatch("prefill batch cannot be empty".into()));
        }
        let model_id = loaded.info.manifest.id.clone();
        let mut events = Vec::with_capacity(requests.len());
        let sequences = requests
            .into_iter()
            .enumerate()
            .map(|(row, (request, execution_sampling))| {
                let leased = cohort.map(|cohort| cohort.take(request.session_id)).transpose()?;
                let restored = restore_prefix(loaded, &request, leased)?;
                let sequence = Sequence::prepare(loaded, request, execution_sampling, restored)?;
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
        if cohort.is_none() && loaded.prefixes.reserve_batch_slots(sequences.len()) {
            crate::engine::clear_memory_cache()?;
        }
        let page_size = loaded.stream.config().kv_cache.block_size.max(1);
        let required_pages = sequences
            .iter()
            .map(|sequence| {
                required_prefill_pages(
                    sequence.request.prompt_tokens.len(),
                    sequence.position,
                    page_size,
                )
            })
            .sum();
        loaded.reserve_prefill_pages(required_pages)?;
        Ok((
            Self {
                model_id,
                inner: Arc::new(Mutex::new(Some(Batch {
                    sequences,
                    cursor: 0,
                    workspace_constrained: false,
                }))),
            },
            events,
        ))
    }

    pub(in crate::native) fn execute_step(
        &self,
        loaded: &mut LoadedModel,
        token_budget: usize,
    ) -> Result<PrefillStep> {
        let mut guard = self.inner.lock()?;
        let batch = guard
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill batch was finished".into()))?;
        let result = batch.execute_step(loaded, token_budget.max(1));
        drop(guard);
        result
    }

    pub(in crate::native) fn finish(&self) -> Result<Vec<FinishedPrefill>> {
        let mut guard = self.inner.lock()?;
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
}

impl Batch {
    fn execute_step(&mut self, loaded: &mut LoadedModel, mut budget: usize) -> Result<PrefillStep> {
        let mut events = Vec::new();
        while budget > 0 && self.sequences.iter().any(|sequence| sequence.output.is_none()) {
            if let Some((used, packed_events)) = self.execute_packed(loaded, budget)? {
                budget -= used;
                events.extend(packed_events);
                continue;
            }
            let row = self.cursor % self.sequences.len();
            self.cursor = (self.cursor + 1) % self.sequences.len();
            let sequence = &mut self.sequences[row];
            if sequence.output.is_some() {
                continue;
            }
            let scalar_budget =
                LoadedModel::pressure_bounded_prefill_budget(budget, self.workspace_constrained)?;
            let used = sequence.advance(loaded, scalar_budget)?;
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

    fn finish(self) -> Result<Vec<FinishedPrefill>> {
        self.sequences
            .into_iter()
            .map(|sequence| {
                let native = sequence.output.ok_or_else(|| {
                    Error::InvalidPrefillBatch("prefill batch is incomplete".into())
                })?;
                Ok(FinishedPrefill {
                    request: sequence.request,
                    native,
                    started: sequence.started,
                })
            })
            .collect()
    }
}
