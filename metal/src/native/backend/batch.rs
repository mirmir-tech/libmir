use std::time::Instant;

use runtime::{
    Result as RuntimeResult,
    backend::{DecodeBatchOutput, DecodeBatchRequest, DecodeOutput, TokenEvent},
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
        self.with_model(&lookup, move |loaded| {
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
            let outputs = native
                .into_iter()
                .zip(&sequences)
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
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            tracing::debug!(
                backend = "metal",
                rows = inputs.len(),
                batched,
                execution_ms = started.elapsed().as_secs_f64() * 1_000.0,
                "completed Metal decode batch"
            );
            Ok(outputs)
        })
    }
}
