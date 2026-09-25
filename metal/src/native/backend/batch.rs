use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{DecodeBatchOutput, DecodeBatchRequest, DecodeOutput, DecodeTimings, TokenEvent},
};

use super::{MetalBackend, execution::execution_sampling};
use crate::native::{
    error::Result,
    model::{DecodeExecution, DecodeInput},
    output,
};

impl MetalBackend {
    pub fn decode_batch_tokens(
        &self,
        request: &DecodeBatchRequest,
    ) -> RuntimeResult<DecodeBatchOutput> {
        DecodeBatchOutput::new(self.decode_batch_inner(request)?)
    }

    fn decode_batch_inner(&self, request: &DecodeBatchRequest) -> Result<Vec<DecodeOutput>> {
        let started = Instant::now();
        let lookup = request.model().id.clone();
        let sequences = request.sequences().to_vec();
        let device_pipeline = self.config.fusion.device_token_pipeline.enabled();
        let profile = self.profile_decode.load(std::sync::atomic::Ordering::Relaxed);
        self.with_model(&lookup, move |loaded| {
            execute_loaded_decode(loaded, &sequences, device_pipeline, profile, started)
        })
    }
}

pub(super) fn execute_loaded_decode(
    loaded: &mut crate::native::model::LoadedModel,
    sequences: &[runtime::backend::DecodeSequence],
    device_pipeline: bool,
    profile: bool,
    started: Instant,
) -> Result<Vec<DecodeOutput>> {
    let executing = Instant::now();
    let inputs = sequences
        .iter()
        .map(|sequence| DecodeInput {
            session: sequence.session_id,
            token: sequence.token_id,
            sampling: execution_sampling(sequence.sampling_logits.clone(), device_pipeline),
        })
        .collect::<Vec<_>>();
    let native = loaded.decode_grouped(&inputs)?;
    let packed_rows = native
        .iter()
        .filter(|(_, mode)| matches!(mode, DecodeExecution::Packed { .. }))
        .count();
    let outputs = native
        .into_iter()
        .zip(sequences)
        .map(|((native, execution), sequence)| {
            let output = output::materialize(loaded, native, sequence.sampling_logits.clone())?;
            Ok(DecodeOutput {
                event: TokenEvent {
                    token_id: output.next_token,
                    text: match execution {
                        DecodeExecution::Packed { .. } => {
                            "metal.decode=packed-device-token-pipeline".into()
                        },
                        DecodeExecution::Scalar => "metal.decode=scalar".into(),
                    },
                    finished: false,
                },
                logits: output.logits,
                candidates: output.candidates,
                timings: profile.then(|| DecodeTimings {
                    batch_rows: match execution {
                        DecodeExecution::Scalar => 1,
                        DecodeExecution::Packed { rows } => rows,
                    },
                    ..DecodeTimings::default()
                }),
            })
        })
        .collect::<Result<Vec<_>>>();
    let mut outputs = loaded.finish_decode(&inputs, outputs)?;
    let finished = Instant::now();
    let timing = super::decode_timing::finish(started, executing, finished);
    for output in &mut outputs {
        if let Some(row) = &mut output.timings {
            row.backend_wait = timing.backend_wait;
            row.backend_execution = timing.backend_execution;
        }
    }
    tracing::trace!(
        backend = "metal",
        rows = inputs.len(),
        packed_rows,
        backend_wait_ms = timing.backend_wait.as_secs_f64() * 1_000.0,
        execution_ms = timing.backend_execution.as_secs_f64() * 1_000.0,
        total_ms = finished.duration_since(started).as_secs_f64() * 1_000.0,
        "completed Metal decode batch"
    );
    Ok(outputs)
}
