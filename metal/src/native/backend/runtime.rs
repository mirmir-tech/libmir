use async_trait::async_trait;
use foundation::model::ModelManifest;
use runtime::{
    Result as RuntimeResult, RuntimeError,
    backend::{
        Backend, BackendCapability, BackendInfo, DecodeOutput, DecodeRequest, GenerationRequest,
        ModelHandle, PrefillOutput, PrefillRequest, TokenEvent,
    },
    trace::ModelTrace,
};

use super::MetalBackend;
use crate::{MetalProgressEvent, native::trace};

#[async_trait]
impl Backend for MetalBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "mlx-native".into(),
            device: "apple-silicon-gpu".into(),
            capabilities: vec![
                BackendCapability::Prefill,
                BackendCapability::Decode,
                BackendCapability::Streaming,
                BackendCapability::PrefixCache,
                BackendCapability::ContinuousBatching,
                BackendCapability::Quantization("int4 affine weights".into()),
            ],
        }
    }

    async fn load_model(&self, manifest: &ModelManifest) -> RuntimeResult<ModelHandle> {
        Ok(self.load_model_inner(manifest, None)?)
    }

    async fn model_trace(&self, model: &ModelHandle) -> RuntimeResult<ModelTrace> {
        let backend = self.info();
        Ok(self.with_model(&model.id, move |loaded| Ok(trace::build(loaded, backend)))?)
    }

    async fn prefill(&self, request: PrefillRequest) -> RuntimeResult<PrefillOutput> {
        let mut ignored = |_event: MetalProgressEvent| {};
        Ok(self.prefill_tokens_inner(
            &request.model,
            request.session_id,
            &request.prompt_tokens,
            &request.block_table,
            request.sampling_logits,
            &mut ignored,
        )?)
    }

    async fn decode(&self, request: DecodeRequest) -> RuntimeResult<DecodeOutput> {
        Ok(self.decode_token_inner(
            &request.model,
            request.session_id,
            request.token_id,
            &request.block_table,
            request.sampling_logits,
        )?)
    }

    async fn generate(
        &self,
        _model: &ModelHandle,
        _request: GenerationRequest,
    ) -> RuntimeResult<Vec<TokenEvent>> {
        Err(RuntimeError::BackendUnavailable(
            "use prefill/decode for native MLX streaming".into(),
        ))
    }
}
