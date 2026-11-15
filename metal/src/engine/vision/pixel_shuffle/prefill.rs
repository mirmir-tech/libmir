use models::vision::{PixelShufflePreprocessedImage, PixelShufflePromptTokens};

use super::PixelShuffleVisionTower;
use crate::engine::{Array, DecoderCache, DecoderModel, Error, ImageTokenSpan, Result, Stream};

impl PixelShuffleVisionTower {
    /// Encodes the image and prefills the prompt with its soft tokens over the
    /// image span; the decoder keeps plain positions.
    pub(crate) fn forward_multimodal_prefill(
        &self,
        decoder: &DecoderModel,
        prompt: &PixelShufflePromptTokens,
        image: &PixelShufflePreprocessedImage,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        let span =
            ImageTokenSpan::new(prompt.image_start..prompt.image_end, prompt.token_ids.len())?;
        if span.len() != image.soft_tokens {
            return Err(Error::InvalidModel(format!(
                "pixel-shuffle vision prompt has {} image tokens but preprocessing produced {}",
                span.len(),
                image.soft_tokens
            )));
        }
        let sequence = i32::try_from(prompt.token_ids.len())?;
        let token_ids = Array::from_u32(&prompt.token_ids, &[1, sequence])?;
        let embeddings = self.forward_preprocessed(image, stream)?;
        decoder.forward_embedded_multimodal(&token_ids, &embeddings, span, cache, stream)
    }
}
