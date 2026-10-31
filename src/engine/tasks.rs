use runtime::{Result, backend::ModelHandle};

use super::{Engine, EngineInner};

impl Engine {
    pub(crate) fn embed_tokens(
        &self,
        model: &ModelHandle,
        inputs: &[Vec<u32>],
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>> {
        #[cfg(not(feature = "metal"))]
        let _ = (model, inputs, dimensions);
        match &self.inner {
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.embed_tokens(model, inputs, dimensions),
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.embed_tokens(model, inputs, dimensions)?),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }

    pub(crate) fn score_tokens(
        &self,
        model: &ModelHandle,
        inputs: &[Vec<u32>],
    ) -> Result<Vec<f32>> {
        #[cfg(not(feature = "metal"))]
        let _ = (model, inputs);
        match &self.inner {
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.score_tokens(model, inputs),
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.score_tokens(model, inputs)?),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }
}

impl Engine {
    /// Whether top-p without a top-k bound samples on the device. Elsewhere
    /// such requests read full logits and sample on the host.
    pub(crate) const fn samples_unbounded_nucleus(&self) -> bool {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => true,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => false,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => false,
        }
    }
}
