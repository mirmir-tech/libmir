use models::vision::{PixelShufflePreprocessedImage, PixelShufflePromptTokens};
use runtime::{
    backend::{ModelHandle, PrefillOutput, SamplingLogits},
    kv::BlockTable,
    progress::ProgressEvent,
};
use uuid::Uuid;

use super::super::{Engine, EngineInner};

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_pixel_shuffle_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &PixelShufflePromptTokens,
        image: &PixelShufflePreprocessedImage,
        block_table: &BlockTable,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> runtime::Result<PrefillOutput> {
        match &self.inner {
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.prefill_pixel_shuffle_vision_with_progress(
                model, session_id, prompt, image, block_table, sampling, progress,
            ),
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => Ok(cuda.prefill_pixel_shuffle_vision_with_progress(
                model, session_id, prompt, image, block_table, sampling, progress,
            )?),
        }
    }
}
