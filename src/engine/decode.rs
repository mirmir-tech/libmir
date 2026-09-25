#[cfg(feature = "cuda")]
use runtime::backend::DecodeRequest;
use runtime::{
    Result as RuntimeResult,
    backend::{DecodeOutput, ModelHandle, SamplingLogits},
    kv::BlockTable,
};
use uuid::Uuid;

#[cfg(not(any(feature = "cuda", feature = "metal")))]
use super::unavailable;
use super::{Engine, EngineInner};

impl Engine {
    /// Decodes one token for a session and returns the next-token prediction.
    pub fn decode_token(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        token_id: u32,
        block_table: &BlockTable,
        sampling: SamplingLogits,
    ) -> RuntimeResult<DecodeOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (&model, session_id, token_id, &block_table);
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        drop(sampling);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.decode_token(&DecodeRequest {
                model: model.clone(),
                session_id,
                token_id,
                block_table: block_table.clone(),
                sampling_logits: sampling,
            })?),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                metal.decode_token(model, session_id, token_id, block_table, sampling)
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }
}
