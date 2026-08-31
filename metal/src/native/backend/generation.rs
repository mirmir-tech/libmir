use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{DecodeBatchRequest, DecodeOutput, PrefillOutput, PrefillRequest},
};

use super::{
    MetalBackend, batch::execute_loaded_decode, execution::execution_sampling,
    prefill_output::materialize_prefill,
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
            Ok(prefill_schedule(routed))
        })?)
    }

    pub fn prepare_prefill_batch(
        &self,
        requests: &[PrefillRequest],
        cohort: Option<&MetalPrefillCohort>,
        progress: &mut dyn FnMut(usize, MetalProgressEvent),
    ) -> RuntimeResult<MetalPrefillBatch> {
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
        let (batch, events) = self.with_model(&lookup, move |loaded| {
            MetalPrefillBatch::prepare(loaded, requests, cohort.as_ref())
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
                |batch| batch.execute_step(loaded, prefill_budget),
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
        Ok(self.with_model(&model_id, move |loaded| {
            batch
                .finish()?
                .into_iter()
                .map(|finished| {
                    materialize_prefill(
                        loaded,
                        &finished.request,
                        finished.native,
                        finished.started,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?)
    }
}

fn validate_prefill_requests(requests: &[PrefillRequest]) -> Result<&PrefillRequest> {
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

const fn prefill_schedule(routed: bool) -> MetalPrefillSchedule {
    if routed {
        MetalPrefillSchedule {
            max_wave_rows: usize::MAX,
            interleave_decode: false,
        }
    } else {
        MetalPrefillSchedule {
            max_wave_rows: usize::MAX,
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
mod tests {
    use super::{prefill_schedule, step_model_id};

    #[test]
    fn empty_generation_step_is_rejected() {
        assert!(step_model_id(None, None).is_err());
    }

    #[test]
    fn routed_prefill_uses_the_scheduler_capacity_before_streaming() {
        let routed = prefill_schedule(true);
        assert_eq!(routed.max_wave_rows, usize::MAX);
        assert!(!routed.interleave_decode);

        let dense = prefill_schedule(false);
        assert_eq!(dense.max_wave_rows, usize::MAX);
        assert!(dense.interleave_decode);
    }
}
