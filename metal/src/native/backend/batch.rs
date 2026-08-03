use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{DecodeBatchOutput, DecodeBatchRequest, DecodeOutput, DecodeTimings, TokenEvent},
};

use super::{MetalBackend, execution::execution_sampling};
use crate::native::{
    error::Result,
    model::{DecodeInput, NativeOutput},
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
    let inputs = sequences
        .iter()
        .map(|sequence| DecodeInput {
            session: sequence.session_id,
            token: sequence.token_id,
            sampling: execution_sampling(sequence.sampling_logits, device_pipeline),
        })
        .collect::<Vec<_>>();
    let batched = loaded.can_decode_batch(&inputs);
    let native = if batched {
        loaded.decode_batch(&inputs)?
    } else {
        inputs
            .iter()
            .map(|input| loaded.decode(input.session, input.token, input.sampling))
            .collect::<Result<Vec<NativeOutput>>>()?
    };
    let mut outputs = native
        .into_iter()
        .zip(sequences)
        .map(|(native, sequence)| {
            let output = output::materialize(loaded, native, sequence.sampling_logits)?;
            Ok(DecodeOutput {
                event: TokenEvent {
                    token_id: output.next_token,
                    text: if batched {
                        "metal.decode=packed-device-token-pipeline".into()
                    } else {
                        "metal.decode=scalar-device-token-replay".into()
                    },
                    finished: false,
                },
                logits: output.logits,
                candidates: output.candidates,
                timings: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if profile {
        let timings = DecodeTimings {
            backend_execution: started.elapsed(),
            batch_rows: inputs.len(),
            ..DecodeTimings::default()
        };
        for output in &mut outputs {
            output.timings = Some(timings);
        }
    }
    tracing::trace!(
        backend = "metal",
        rows = inputs.len(),
        batched,
        execution_ms = started.elapsed().as_secs_f64() * 1_000.0,
        "completed Metal decode batch"
    );
    Ok(outputs)
}
