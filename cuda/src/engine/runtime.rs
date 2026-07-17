use async_trait::async_trait;
use foundation::model::ModelManifest;
use runtime::{
    Result as RuntimeResult, RuntimeError,
    backend::{
        Backend, BackendCapability, BackendInfo, DecodeBatchOutput, DecodeBatchRequest,
        DecodeOutput, DecodeRequest, GenerationRequest, ModelHandle, PrefillOutput, PrefillRequest,
        TokenEvent,
    },
    trace::ModelTrace,
};

use super::CudaEngine;

#[async_trait]
impl Backend for CudaEngine {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "cuda-native".into(),
            device: self.backend.device_info().name.clone(),
            capabilities: vec![
                BackendCapability::Prefill,
                BackendCapability::Decode,
                BackendCapability::Streaming,
                BackendCapability::PrefixCache,
                BackendCapability::GraphCapture,
                BackendCapability::ContinuousBatching,
                BackendCapability::Quantization("ModelOpt NVFP4".into()),
            ],
        }
    }

    async fn load_model(&self, manifest: &ModelManifest) -> RuntimeResult<ModelHandle> {
        let mut ignored = |_event| {};
        Ok(self.load_model_with_progress(manifest, &mut ignored)?)
    }

    async fn model_trace(&self, model: &ModelHandle) -> RuntimeResult<ModelTrace> {
        let loaded = self.model(&model.id)?;
        Ok(super::trace::build(loaded.as_ref(), self.info(), self.cache)?)
    }

    async fn prefill(&self, request: PrefillRequest) -> RuntimeResult<PrefillOutput> {
        let mut ignored = |_event| {};
        Ok(self.prefill_with_progress(&request, &mut ignored)?)
    }

    async fn decode(&self, request: DecodeRequest) -> RuntimeResult<DecodeOutput> {
        Ok(self.decode_token(&request)?)
    }

    async fn decode_batch(&self, request: DecodeBatchRequest) -> RuntimeResult<DecodeBatchOutput> {
        Ok(self.decode_batch_tokens(&request)?)
    }

    async fn generate(
        &self,
        model: &ModelHandle,
        _request: GenerationRequest,
    ) -> RuntimeResult<Vec<TokenEvent>> {
        Err(RuntimeError::BackendUnavailable(format!(
            "use prefill/decode streaming for CUDA model {}",
            model.id
        )))
    }
}
