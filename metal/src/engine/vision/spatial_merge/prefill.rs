use models::vision::{SpatialMergePreprocessedImage, SpatialMergePromptTokens};

use super::SpatialMergeVisionTower;
use crate::engine::{
    Array, DecoderCache, Error, ImageTokenSpan, Result, Stream,
    hybrid_linear_moe::HybridLinearMoeModel,
};

impl SpatialMergeVisionTower {
    pub fn forward_multimodal_prefill(
        &self,
        decoder: &HybridLinearMoeModel,
        prompt: &SpatialMergePromptTokens,
        image: &SpatialMergePreprocessedImage,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        let span =
            ImageTokenSpan::new(prompt.image_start..prompt.image_end, prompt.token_ids.len())?;
        if span.len() != image.soft_tokens {
            return Err(Error::InvalidModel(format!(
                "spatial-merge vision prompt has {} image tokens but preprocessing produced {}",
                span.len(),
                image.soft_tokens
            )));
        }
        let sequence = i32::try_from(prompt.token_ids.len())?;
        let token_ids = Array::from_u32(&prompt.token_ids, &[1, sequence])?;
        let position_ids = Array::from_u32(&prompt.position_ids, &[3, sequence])?;
        let embeddings = self.forward_preprocessed(image, stream)?;
        decoder
            .forward_multimodal_prefill(&token_ids, &embeddings, span, &position_ids, cache, stream)
    }
}
