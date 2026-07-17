#[cfg(any(feature = "cuda", feature = "metal"))]
use runtime::backend::DecodeBatchRequest;
use runtime::backend::{DecodeOutput, DecodeSequence, ModelHandle};

use super::{Engine, EngineInner};
use crate::Result;

impl Engine {
    pub(crate) fn decode_sequences(
        &self,
        model: &ModelHandle,
        sequences: Vec<DecodeSequence>,
    ) -> Result<Vec<DecodeOutput>> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = model;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        drop(sequences);
        #[cfg(any(feature = "cuda", feature = "metal"))]
        if sequences.len() == 1 {
            let sequence = &sequences[0];
            return Ok(vec![self.decode_token(
                model,
                sequence.session_id,
                sequence.token_id,
                &sequence.block_table,
                sequence.sampling_logits,
            )?]);
        }
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda
                .decode_batch_tokens(&DecodeBatchRequest::new(model.clone(), sequences)?)?
                .into_outputs()),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => Ok(metal
                .decode_batch_tokens(&DecodeBatchRequest::new(model.clone(), sequences)?)?
                .into_outputs()),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Err(runtime::RuntimeError::BackendUnavailable(
                "libmir was built without an accelerator feature".into(),
            )
            .into()),
        }
    }
}
