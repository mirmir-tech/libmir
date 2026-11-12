//! Preparing one image for a pixel-shuffle encoder: the placeholder becomes
//! the image token, the image is sized to the patch budget the attention
//! limit leaves, and the image token expands to the image tags around one
//! token per soft token.

use foundation::conversation::Conversation;
use models::{
    generation::GenerationSettings,
    layout::{PixelShuffleImageProcessorConfig, PixelShuffleVisionConfig},
    vision::PixelShufflePromptTokens,
};

use super::{PreparedVisionPrompt, limits::VisionLimits};
use crate::{
    Result,
    model::{ModelDescriptor, helpers::validate_context},
};

impl ModelDescriptor {
    pub(super) fn prepare_pixel_shuffle(
        &self,
        conversation: &Conversation,
        encoded_image: &[u8],
        generation: GenerationSettings,
        limits: Option<VisionLimits>,
        vision: &PixelShuffleVisionConfig,
        processor: &PixelShuffleImageProcessorConfig,
    ) -> Result<PreparedVisionPrompt> {
        let request = self.with_image_token(conversation, vision.image_token_id)?;
        let prepared = self.prepare_with_settings(&request, generation)?;
        let image = match limits {
            Some(limits) => processor.preprocess_encoded_with_patch_limit(
                encoded_image,
                limits.pixel_shuffle_patch_limit(vision),
            )?,
            None => processor.preprocess_encoded(encoded_image)?,
        };
        if let Some(limits) = limits {
            // The encoder attends over its prefix tokens and every patch.
            limits.validate(
                vision.prefix_tokens + image.grid_height * image.grid_width,
                vision.num_attention_heads,
            )?;
        }
        let tags = self.image_tag_ids(vision)?;
        let tokens = PixelShufflePromptTokens::prepare(
            &prepared.tokens.token_ids,
            &image,
            vision.image_token_id,
            tags,
        )?;
        validate_context(tokens.token_ids.len(), generation.max_tokens, self.metadata.context_len)?;
        Ok(PreparedVisionPrompt::PixelShuffle { prompt: prepared.prompt, tokens, image })
    }

    /// The token ids of the image tags the configuration names.
    pub(crate) fn image_tag_ids(&self, vision: &PixelShuffleVisionConfig) -> Result<(u32, u32)> {
        let id = |tag: &str| {
            self.tokenizer.token_id(tag).ok_or_else(|| {
                models::ModelsError::InvalidConfig(format!("image tag {tag} is not one token"))
            })
        };
        Ok((id(vision.image_tags.start())?, id(vision.image_tags.end())?))
    }
}
