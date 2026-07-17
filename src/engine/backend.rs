use async_trait::async_trait;
use foundation::model::ModelManifest;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use runtime::RuntimeError;
use runtime::{
    Result as RuntimeResult,
    backend::{
        Backend, BackendInfo, DecodeBatchOutput, DecodeBatchRequest, DecodeOutput, DecodeRequest,
        GenerationRequest, ModelHandle, PrefillOutput, PrefillRequest, TokenEvent,
    },
    trace::ModelTrace,
};

use super::{Engine, EngineInner};

#[async_trait]
impl Backend for Engine {
    fn info(&self) -> BackendInfo {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.info(),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.info(),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => BackendInfo {
                name: "unavailable".into(),
                device: "none".into(),
                capabilities: Vec::new(),
            },
        }
    }

    async fn load_model(&self, manifest: &ModelManifest) -> RuntimeResult<ModelHandle> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = manifest;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.load_model(manifest).await,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.load_model(manifest).await,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    async fn model_trace(&self, model: &ModelHandle) -> RuntimeResult<ModelTrace> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = model;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.model_trace(model).await,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.model_trace(model).await,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    async fn prefill(&self, request: PrefillRequest) -> RuntimeResult<PrefillOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = &request;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.prefill(request).await,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.prefill(request).await,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    async fn decode(&self, request: DecodeRequest) -> RuntimeResult<DecodeOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = &request;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.decode(request).await,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.decode(request).await,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    async fn decode_batch(&self, request: DecodeBatchRequest) -> RuntimeResult<DecodeBatchOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = &request;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.decode_batch(request).await,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.decode_batch(request).await,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }

    async fn generate(
        &self,
        model: &ModelHandle,
        request: GenerationRequest,
    ) -> RuntimeResult<Vec<TokenEvent>> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = (model, &request);
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.generate(model, request).await,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.generate(model, request).await,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => unavailable(),
        }
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn unavailable<T>() -> RuntimeResult<T> {
    Err(RuntimeError::BackendUnavailable(
        "libmir was built without an accelerator feature".into(),
    ))
}
