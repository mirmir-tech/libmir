use std::time::Instant;

use runtime::{
    backend::{DecodeBatchRequest, DecodeOutput},
    progress::ProgressEvent,
};

use super::CudaPrefillBatch;
use crate::{
    Result,
    engine::{CudaEngine, profile::DecodeProfile},
};

pub struct CudaGenerationStepOutput {
    pub decode: Vec<DecodeOutput>,
    pub prefill: Result<bool>,
}

impl CudaEngine {
    pub fn execute_generation_step(
        &self,
        decode: Option<&DecodeBatchRequest>,
        prefill: Option<&mut CudaPrefillBatch>,
        prefill_budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<CudaGenerationStepOutput> {
        let model_id = step_model_id(decode, prefill.as_deref())?;
        let loaded = self.model(model_id)?;
        if let Some(request) = decode {
            for sequence in request.sequences() {
                loaded.require_session(sequence.session_id)?;
            }
        }
        let waiting = Instant::now();
        let mut runner = loaded.prefill_runner()?;
        let wait = waiting.elapsed();
        let profile = DecodeProfile::begin(
            &self.backend,
            wait,
            decode.map_or(0, |request| request.sequences().len()),
            self.profile_decode() && decode.is_some(),
        )?;
        let interleaved_decode = decode.is_some();
        let decode_started = Instant::now();
        let mut outputs = match decode {
            Some(request) => self.decode_batch_with_runner(&mut runner, request)?,
            None => Vec::new(),
        };
        let decode_execution = decode_started.elapsed();
        let prefill = prefill.map_or_else(
            || Ok(true),
            |batch| {
                self.execute_prefill_batch_step_with_runner(
                    batch,
                    prefill_budget,
                    interleaved_decode,
                    progress,
                    &mut runner,
                    wait,
                )
            },
        );
        drop(runner);
        if let Some(profile) = profile {
            profile.finish(&self.backend, &mut outputs)?;
        }
        if let Some(request) = decode {
            tracing::debug!(
                backend = "cuda",
                rows = request.sequences().len(),
                runner_wait_ms = wait.as_secs_f64() * 1_000.0,
                execution_ms = decode_execution.as_secs_f64() * 1_000.0,
                "completed CUDA generation-step decode"
            );
        }
        Ok(CudaGenerationStepOutput { decode: outputs, prefill })
    }
}

fn step_model_id<'a>(
    decode: Option<&'a DecodeBatchRequest>,
    prefill: Option<&'a CudaPrefillBatch>,
) -> Result<&'a str> {
    let decode_id = decode.map(|request| request.model().id.as_str());
    let prefill_id = prefill.map(CudaEngine::prefill_batch_model_id);
    match (decode_id, prefill_id) {
        (Some(decode), Some(prefill)) if decode != prefill => {
            Err(crate::Error::State("CUDA generation step targets multiple models".into()))
        },
        (Some(model), _) | (_, Some(model)) => Ok(model),
        (None, None) => Err(crate::Error::State("CUDA generation step is empty".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::step_model_id;

    #[test]
    fn empty_generation_step_is_rejected() {
        assert!(step_model_id(None, None).is_err());
    }
}
