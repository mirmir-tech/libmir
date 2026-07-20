use models::vision::{SpatialMergePreprocessedImage, SpatialMergePromptTokens};
use runtime::{
    backend::{ModelHandle, PrefillOutput, SamplingLogits},
    kv::BlockTable,
    progress::ProgressEvent,
};
use uuid::Uuid;

#[cfg(feature = "metal")]
use super::metal_progress;
use super::{Engine, EngineInner};

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_spatial_merge_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &SpatialMergePromptTokens,
        image: &SpatialMergePreprocessedImage,
        block_table: &BlockTable,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> runtime::Result<PrefillOutput> {
        match &self.inner {
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let mut mapped = |event| progress(metal_progress(event));
                metal.prefill_spatial_merge_vision_with_progress(
                    model, session_id, prompt, image, block_table, sampling, &mut mapped,
                )
            },
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.prefill_spatial_merge_vision_with_progress(
                model, session_id, prompt, image, block_table, sampling, progress,
            )?),
        }
    }
}
