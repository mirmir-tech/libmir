use std::sync::Arc;

#[cfg(any(feature = "cuda", feature = "metal"))]
use foundation::model::BackendTarget;
use runtime::{
    backend::{DecodeOutput, DecodeSequence, ModelHandle, PrefillOutput, PrefillRequest},
    kv::CacheConfig,
    progress::ProgressEvent,
    scheduler::SchedulerConfig,
};

#[cfg(any(feature = "cuda", feature = "metal"))]
use super::generation::GenerationCoordinator;
#[cfg(any(feature = "cuda", feature = "metal"))]
use super::response::DecodeResponse;
use super::{DecodeCoordinator, GenerationStepState, PrefillCoordinator};
use crate::{Engine, Result};

pub struct ModelCoordinator {
    inner: Coordinator,
}

enum Coordinator {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Generation(GenerationCoordinator),
    Split {
        decode: Box<DecodeCoordinator>,
        prefill: Box<PrefillCoordinator>,
    },
}

pub struct PendingModelDecode(PendingModelDecodeInner);

enum PendingModelDecodeInner {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Generation(Arc<DecodeResponse>),
    Split(DecodeSequence),
}

impl ModelCoordinator {
    #[cfg_attr(
        not(any(feature = "cuda", feature = "metal")),
        expect(
            clippy::unnecessary_wraps,
            reason = "accelerator builds can fail while starting the dedicated generation worker"
        )
    )]
    pub(crate) fn new(
        engine: Engine,
        model: ModelHandle,
        config: SchedulerConfig,
        cache: CacheConfig,
    ) -> Result<Self> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        if uses_generation_worker(&engine.target()) {
            return Ok(Self {
                inner: Coordinator::Generation(GenerationCoordinator::new(
                    engine, model, config, cache,
                )?),
            });
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = cache;
        let step =
            Arc::new(GenerationStepState::new(config.max_batch_tokens, config.max_batch_requests));
        Ok(Self {
            inner: Coordinator::Split {
                decode: Box::new(DecodeCoordinator::new(
                    engine.clone(),
                    model.clone(),
                    config.clone(),
                    step.clone(),
                )),
                prefill: Box::new(PrefillCoordinator::new(engine, model, config, step)),
            },
        })
    }

    #[cfg_attr(
        not(any(feature = "cuda", feature = "metal")),
        expect(
            clippy::unnecessary_wraps,
            reason = "accelerator generation workers can reject decode admission"
        )
    )]
    pub(crate) fn start_decode(&self, sequence: DecodeSequence) -> Result<PendingModelDecode> {
        match &self.inner {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Coordinator::Generation(coordinator) => Ok(PendingModelDecode(
                PendingModelDecodeInner::Generation(coordinator.start_decode(sequence)?),
            )),
            Coordinator::Split { .. } => {
                Ok(PendingModelDecode(PendingModelDecodeInner::Split(sequence)))
            },
        }
    }

    pub(crate) fn finish_decode(&self, pending: PendingModelDecode) -> Result<DecodeOutput> {
        match (&self.inner, pending.0) {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            (Coordinator::Generation(_), PendingModelDecodeInner::Generation(response)) => {
                response.wait()
            },
            (Coordinator::Split { decode, .. }, PendingModelDecodeInner::Split(sequence)) => {
                decode.submit(sequence)
            },
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(runtime::RuntimeError::Scheduler(
                "decode continuation targets another coordinator".into(),
            )
            .into()),
        }
    }

    pub(crate) fn submit_prefill(
        &self,
        request: PrefillRequest,
        expects_decode: bool,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = expects_decode;
        match &self.inner {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Coordinator::Generation(coordinator) => {
                coordinator.submit_prefill(request, expects_decode, progress)
            },
            Coordinator::Split { prefill, .. } => prefill.submit(request, progress),
        }
    }

    pub(crate) fn release(&self, session_id: uuid::Uuid) {
        match &self.inner {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Coordinator::Generation(coordinator) => coordinator.release(session_id),
            Coordinator::Split { decode, .. } => decode.release(session_id),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn uses_generation_worker(target: &BackendTarget) -> bool {
    match target {
        #[cfg(feature = "cuda")]
        BackendTarget::Cuda => true,
        #[cfg(feature = "metal")]
        BackendTarget::Metal => true,
        _ => false,
    }
}
