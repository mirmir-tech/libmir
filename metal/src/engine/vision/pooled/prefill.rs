use models::vision::{PooledPreprocessedImage, PooledPromptTokens};

use super::PooledVisionTower;
use crate::engine::{
    Array, DecoderCache, Error, ImageTokenSpan, Result, Stream, hybrid_moe::HybridMoeModel,
};

impl PooledVisionTower {
    /// Encodes one image, replaces its expanded placeholder embeddings, and
    /// runs the initial decoder prefill while all embeddings remain on Metal.
    pub fn forward_multimodal_prefill(
        &self,
        decoder: &HybridMoeModel,
        prompt: &PooledPromptTokens,
        image: &PooledPreprocessedImage,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        let span =
            ImageTokenSpan::new(prompt.image_start..prompt.image_end, prompt.token_ids.len())?;
        if span.len() != image.soft_tokens {
            return Err(Error::InvalidModel(format!(
                "pooled vision prompt has {} image tokens but preprocessing produced {}",
                span.len(),
                image.soft_tokens
            )));
        }
        let sequence = i32::try_from(prompt.token_ids.len())?;
        let token_ids = Array::from_u32(&prompt.token_ids, &[1, sequence])?;
        let embeddings = self.forward_preprocessed(image, stream)?;
        decoder.forward_multimodal_prefill(
            &token_ids,
            &embeddings,
            span,
            self.bidirectional_image_attention,
            cache,
            stream,
        )
    }
}
