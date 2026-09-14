mod cancellation;

use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{DecodeBatchRequest, DecodeOutput, PrefillOutput, PrefillRequest},
};

use super::{
    MetalBackend, batch::execute_loaded_decode, execution::execution_sampling,
    prefill_output::finish_prefill,
};
use crate::{
    MetalProgressEvent,
    native::{
        error::{Error, Result},
        prefill::{MetalPrefillBatch, MetalPrefillCohort, PrefillStep},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalPrefillSchedule {
    pub max_wave_rows: usize,
    pub max_wave_tokens: usize,
    pub max_cohort_tokens: usize,
    pub interleave_decode: bool,
}

pub struct MetalGenerationStepOutput {
    pub decode: Vec<DecodeOutput>,
    pub prefill: RuntimeResult<bool>,
}

impl MetalBackend {
    pub fn prepare_prefill_cohort(
        &self,
        requests: &[PrefillRequest],
    ) -> RuntimeResult<MetalPrefillCohort> {
        let first = validate_prefill_requests(requests)?;
        let lookup = first.model.id.clone();
        let requests = requests.to_vec();
        Ok(self.with_model(&lookup, move |loaded| MetalPrefillCohort::prepare(loaded, &requests))?)
    }

    /// Returns the prompt tokens processed by one graph for `model`.
    pub fn prefill_chunk_tokens(
        &self,
        model: &runtime::backend::ModelHandle,
    ) -> RuntimeResult<usize> {
        let lookup = model.id.clone();
        Ok(self.with_model(&lookup, move |loaded| Ok(loaded.info.prefill_step))?)
    }

    /// Returns the physical wave and streaming policy measured for the loaded
    /// model architecture.
    pub fn prefill_schedule(
        &self,
        model: &runtime::backend::ModelHandle,
    ) -> RuntimeResult<MetalPrefillSchedule> {
        let lookup = model.id.clone();
        Ok(self.with_model(&lookup, move |loaded| {
            let routed = loaded
                .info
                .decoder
                .as_ref()
                .is_some_and(|decoder| decoder.num_experts.is_some_and(|experts| experts > 0));
            let memory_budgets =
                routed.then(|| loaded.prefill_memory_token_budgets()).transpose()?;
            Ok(prefill_schedule(routed, memory_budgets))
        })?)
    }

    pub fn prepare_prefill_batch(
        &self,
        requests: &[PrefillRequest],
        cohort: Option<&MetalPrefillCohort>,
        progress: &mut dyn FnMut(usize, MetalProgressEvent),
    ) -> RuntimeResult<MetalPrefillBatch> {
        let submitted = Instant::now();
        let first = validate_prefill_requests(requests)?;
        if cohort.is_some_and(|cohort| cohort.model_id() != first.model.id) {
            return Err(Error::InvalidPrefillBatch(
                "prefill batch targets another logical cohort".into(),
            )
            .into());
        }
        let lookup = first.model.id.clone();
        let cohort = cohort.cloned();
        let device_pipeline = self.config.fusion.device_token_pipeline.enabled();
        let requests = requests
            .iter()
            .cloned()
            .map(|request| {
                let sampling = execution_sampling(request.sampling_logits, device_pipeline);
                (request, sampling)
            })
            .collect();
        let client = self.model_client(&lookup)?;
        let retire = client.prefill_retirement();
        let (batch, events) = client.run(move |loaded| {
            MetalPrefillBatch::prepare_at(
                loaded,
                requests,
                cohort.as_ref(),
                submitted,
                Some(retire),
            )
        })?;
        for (row, event) in events {
            progress(row, event);
        }
        Ok(batch)
    }

    pub fn execute_generation_step(
        &self,
        decode: Option<&DecodeBatchRequest>,
        prefill: Option<&MetalPrefillBatch>,
        prefill_budget: usize,
        progress: &mut dyn FnMut(usize, MetalProgressEvent),
    ) -> RuntimeResult<MetalGenerationStepOutput> {
        self.execute_generation_step_until(decode, prefill, prefill_budget, progress, || false)
    }

    /// Yields prefill between evaluated graphs when requested, preserving row
    /// indices and the original budget for subsequent continuation.
    pub fn execute_generation_step_until(
        &self,
        decode: Option<&DecodeBatchRequest>,
        prefill: Option<&MetalPrefillBatch>,
        prefill_budget: usize,
        progress: &mut dyn FnMut(usize, MetalProgressEvent),
        mut should_yield: impl FnMut() -> bool + Send + 'static,
    ) -> RuntimeResult<MetalGenerationStepOutput> {
        let model_id = step_model_id(decode, prefill)?.to_owned();
        let sequences = decode.map_or_else(Vec::new, |request| request.sequences().to_vec());
        let batch = prefill.cloned();
        let device_pipeline = self.config.fusion.device_token_pipeline.enabled();
        let profile = self.profile_decode.load(std::sync::atomic::Ordering::Relaxed);
        let started = Instant::now();
        let (decode, prefill) = self.with_model(&model_id, move |loaded| {
            let decode = if sequences.is_empty() {
                Vec::new()
            } else {
                execute_loaded_decode(loaded, &sequences, device_pipeline, profile, started)?
            };
            let prefill = batch.map_or_else(
                || Ok(PrefillStep { events: Vec::new(), complete: true }),
                |batch| batch.execute_step_until(loaded, prefill_budget, &mut should_yield),
            );
            Ok((decode, prefill))
        })?;
        match prefill {
            Ok(step) => {
                for (row, event) in step.events {
                    progress(row, event);
                }
                Ok(MetalGenerationStepOutput { decode, prefill: Ok(step.complete) })
            },
            Err(error) => Ok(MetalGenerationStepOutput { decode, prefill: Err(error.into()) }),
        }
    }

    pub fn finish_prefill_batch(
        &self,
        batch: MetalPrefillBatch,
    ) -> RuntimeResult<Vec<PrefillOutput>> {
        let model_id = batch.model_id().to_owned();
        Ok(self.with_model(&model_id, move |loaded| finish_prefill(loaded, &batch))?)
    }
}

fn validate_prefill_requests(requests: &[PrefillRequest]) -> Result<&PrefillRequest> {
    crate::native::prefill::validation::validate_requests(requests)?;
    let first = requests
        .first()
        .ok_or_else(|| Error::InvalidPrefillBatch("prefill batch cannot be empty".into()))?;
    if requests.iter().any(|request| {
        request.model.id != first.model.id || request.model.backend != first.model.backend
    }) {
        return Err(Error::InvalidPrefillBatch("prefill batch targets multiple models".into()));
    }
    Ok(first)
}

const fn prefill_schedule(
    routed: bool,
    memory_budgets: Option<(usize, usize)>,
) -> MetalPrefillSchedule {
    if routed {
        let (max_wave_tokens, max_cohort_tokens) = match memory_budgets {
            Some(budgets) => budgets,
            None => (1, 1),
        };
        MetalPrefillSchedule {
            max_wave_rows: usize::MAX,
            max_wave_tokens,
            max_cohort_tokens,
            interleave_decode: false,
        }
    } else {
        MetalPrefillSchedule {
            max_wave_rows: usize::MAX,
            max_wave_tokens: usize::MAX,
            max_cohort_tokens: usize::MAX,
            interleave_decode: true,
        }
    }
}

fn step_model_id<'a>(
    decode: Option<&'a DecodeBatchRequest>,
    prefill: Option<&'a MetalPrefillBatch>,
) -> Result<&'a str> {
    let decode_id = decode.map(|request| request.model().id.as_str());
    let prefill_id = prefill.map(MetalPrefillBatch::model_id);
    match (decode_id, prefill_id) {
        (Some(decode), Some(prefill)) if decode != prefill => {
            Err(Error::InvalidPrefillBatch("generation step targets multiple models".into()))
        },
        (Some(model), _) | (_, Some(model)) => Ok(model),
        (None, None) => Err(Error::InvalidPrefillBatch("generation step is empty".into())),
    }
}

#[cfg(test)]
mod tests;
