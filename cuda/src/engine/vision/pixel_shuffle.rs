//! Prefill of one image through a pixel-shuffle encoder: the encoder runs
//! step by step between other sessions' decodes, then its soft tokens
//! replace the image tokens' embeddings in the decoder's prefill. The
//! decoder takes plain positions: the image tokens sit where the prompt
//! puts them, and no axis of the image enters the decoder's positions.

use models::{
    layout::VisionPipeline,
    vision::{PixelShufflePreprocessedImage, PixelShufflePromptTokens},
};
use runtime::{
    backend::{ModelHandle, PrefillOutput, SamplingLogits},
    kv::BlockTable,
    progress::ProgressEvent,
};
use uuid::Uuid;

use super::{
    super::{
        CudaEngine,
        model::{DeviceToken, ModelExecution, SpatialVisionPrefill},
    },
    model::LoadedVisionModel,
    require_vision, submit_vision_step,
};
use crate::{Error, Result};

impl CudaEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_pixel_shuffle_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &PixelShufflePromptTokens,
        image: &PixelShufflePreprocessedImage,
        block_table: &BlockTable,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let loaded = self.model(&model.id)?;
        let LoadedVisionModel::PixelShuffle(tower) =
            require_vision(&loaded, VisionPipeline::PixelShuffleEncoder)?
        else {
            return Err(Error::UnsupportedVisionContract(
                "loaded CUDA vision tower differs from the discovered pixel-shuffle contract"
                    .into(),
            ));
        };
        if prompt.image_start >= prompt.image_end
            || prompt.image_end > prompt.token_ids.len()
            || prompt.image_end - prompt.image_start != image.soft_tokens
        {
            return Err(Error::InvalidVisionKernel(
                "pixel-shuffle prompt image span differs from preprocessing",
            ));
        }
        let hidden = loaded
            .decoder
            .as_ref()
            .ok_or_else(|| Error::State("vision requires a decoder".into()))?
            .hidden_size;
        progress(ProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let encoded = tower
            .forward_preprocessed_scheduled(image, &mut |step| submit_vision_step(&loaded, step))?;
        let expected = image
            .soft_tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidVisionKernel("pixel-shuffle output size overflow"))?;
        if encoded.tokens != image.soft_tokens
            || encoded.width != hidden
            || encoded.hidden.len() != expected
        {
            return Err(Error::InvalidVisionKernel(
                "pixel-shuffle tower output differs from decoder embedding contract",
            ));
        }
        let positions = plain_positions(prompt.token_ids.len())?;
        let mut runner = loaded.prefill_runner()?;
        let ModelExecution::Generation(generation) = &mut runner.execution else {
            return Err(Error::UnsupportedVisionContract(
                "pixel-shuffle vision requires a generation runner".into(),
            ));
        };
        let completed = generation.prefill_spatial_vision(
            &self.backend,
            SpatialVisionPrefill {
                session: session_id,
                tokens: &prompt.token_ids,
                positions: &positions,
                image: &encoded.hidden,
                image_span: (prompt.image_start, prompt.image_end),
                position_delta: 0,
                table: block_table,
                sampling,
            },
        )?;
        drop(encoded);
        runner.selected = completed.token.map(|token| DeviceToken { session: session_id, token });
        drop(runner);
        loaded.register_session(session_id)?;
        progress(ProgressEvent::prefill_tokens(prompt.token_ids.len(), prompt.token_ids.len()));
        Ok(PrefillOutput {
            accepted_tokens: prompt.token_ids.len(),
            next_token: completed.token,
            trace: Some(format!(
                "cuda.pixel-shuffle=device-tower+shared-routed-prefill;layers={}",
                tower.layer_count()
            )),
            logits: completed.logits,
            candidates: None,
            timings: None,
        })
    }
}

/// Text positions on each of the decoder's three position axes.
fn plain_positions(tokens: usize) -> Result<Vec<u32>> {
    let positions = (0..tokens).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(positions.repeat(3))
}
