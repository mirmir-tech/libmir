use runtime::{
    backend::{
        DecodeBatchRequest, DecodeOutput, DecodeSequence, ModelHandle, PrefillOutput,
        PrefillRequest,
    },
    progress::ProgressEvent,
};

use super::{Engine, EngineInner};
use crate::Result;

mod cancellation;
mod cohort;
mod completion;
mod profile;
mod refill;
mod scheduling;
pub use cohort::EnginePrefillCohort;
pub use profile::{PrefillAdmissionPolicy, PrefillExecutionProfile, PrefillRefillPolicy};
pub enum EnginePrefillBatch {
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaPrefillBatch),
    #[cfg(feature = "metal")]
    Metal(metal::MetalPrefillBatch),
}

pub struct EngineGenerationStepOutput {
    pub decode: Vec<DecodeOutput>,
    pub prefill: Result<bool>,
}

impl Engine {
    pub(crate) fn prepare_generation_prefill(
        &self,
        requests: &[PrefillRequest],
        cohort: Option<&EnginePrefillCohort>,
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<EnginePrefillBatch> {
        #[cfg(not(feature = "metal"))]
        let _ = cohort;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                Ok(EnginePrefillBatch::Cuda(cuda.prepare_prefill_batch(requests, progress)?))
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let cohort = cohort.and_then(|cohort| cohort.metal.as_ref());
                Ok(EnginePrefillBatch::Metal(
                    metal.prepare_prefill_batch(requests, cohort, progress)?,
                ))
            },
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn execute_generation_step(
        &self,
        model: &ModelHandle,
        sequences: Vec<DecodeSequence>,
        prefill: Option<&mut EnginePrefillBatch>,
        prefill_budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
        should_yield: impl FnMut() -> bool + Send + 'static,
    ) -> Result<EngineGenerationStepOutput> {
        #[cfg(not(feature = "metal"))]
        drop(should_yield);
        let request = if sequences.is_empty() {
            None
        } else {
            Some(DecodeBatchRequest::new(model.clone(), sequences)?)
        };
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let batch = match prefill {
                    Some(EnginePrefillBatch::Cuda(batch)) => Some(batch),
                    None => None,
                    #[cfg(feature = "metal")]
                    Some(EnginePrefillBatch::Metal(_)) => return Err(batch_backend_mismatch()),
                };
                let output = cuda.execute_generation_step(
                    request.as_ref(),
                    batch,
                    prefill_budget,
                    progress,
                )?;
                Ok(EngineGenerationStepOutput {
                    decode: output.decode,
                    prefill: match output.prefill {
                        Ok(prefill) => Ok(prefill),
                        Err(error) => Err(error.into()),
                    },
                })
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let batch = match prefill {
                    Some(EnginePrefillBatch::Metal(batch)) => Some(batch),
                    None => None,
                    #[cfg(feature = "cuda")]
                    Some(EnginePrefillBatch::Cuda(_)) => return Err(batch_backend_mismatch()),
                };
                let output = metal.execute_generation_step_until(
                    request.as_ref(),
                    batch.as_deref(),
                    prefill_budget,
                    progress,
                    should_yield,
                )?;
                Ok(EngineGenerationStepOutput {
                    decode: output.decode,
                    prefill: match output.prefill {
                        Ok(prefill) => Ok(prefill),
                        Err(error) => Err(error.into()),
                    },
                })
            },
        }
    }

    pub(crate) fn finish_generation_prefill(
        &self,
        batch: EnginePrefillBatch,
    ) -> Result<Vec<PrefillOutput>> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => match batch {
                EnginePrefillBatch::Cuda(batch) => Ok(cuda.finish_prefill_batch(batch)?),
                #[cfg(feature = "metal")]
                EnginePrefillBatch::Metal(_) => Err(batch_backend_mismatch()),
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => match batch {
                EnginePrefillBatch::Metal(batch) => Ok(metal.finish_prefill_batch(batch)?),
                #[cfg(feature = "cuda")]
                EnginePrefillBatch::Cuda(_) => Err(batch_backend_mismatch()),
            },
        }
    }
}

#[cfg(all(feature = "cuda", feature = "metal"))]
fn batch_backend_mismatch() -> crate::Error {
    runtime::RuntimeError::Backend("generation prefill batch targets another backend".into()).into()
}
