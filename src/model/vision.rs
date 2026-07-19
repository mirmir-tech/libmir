use foundation::protocol::ChatCompletionRequest;
use models::{
    generation::GenerationSettings,
    layout::{ImageProcessorConfig, VisionConfig},
    vision::{
        PooledPreprocessedImage, PooledPromptTokens, SpatialMergePreprocessedImage,
        SpatialMergePromptTokens,
    },
};

use super::{Model, ModelDescriptor, vision_limits::VisionLimits};
use crate::{Result, model::helpers::validate_context};

/// Model-independent marker accepted by vision preparation before it is
/// replaced with the checkpoint's actual image token.
pub const IMAGE_PLACEHOLDER: &str = "<|mirmir_image|>";

#[derive(Debug, Clone)]
/// Prepared single-image input carrying the execution contract discovered from
/// checkpoint configuration and tensors.
pub enum PreparedVisionPrompt {
    /// Fixed patch grid followed by spatial pooling and text projection.
    Pooled {
        /// Rendered text prompt before image placeholder expansion.
        prompt: models::chat::ChatPrompt,
        /// Expanded token sequence and exact soft-token span.
        tokens: PooledPromptTokens,
        /// Decoded, resized, and patch-packed image.
        image: PooledPreprocessedImage,
    },
    /// Temporal-spatial patches followed by a learned spatial merger.
    SpatialMerge {
        /// Rendered text prompt before image placeholder expansion.
        prompt: models::chat::ChatPrompt,
        /// Expanded tokens, M-RoPE positions, and decode position delta.
        tokens: SpatialMergePromptTokens,
        /// Decoded, normalized, and temporal-spatial patch buffer.
        image: SpatialMergePreprocessedImage,
    },
}

impl ModelDescriptor {
    /// Prepares one encoded image and expands exactly one image placeholder in
    /// the rendered request.
    pub fn prepare_image(
        &self,
        request: &ChatCompletionRequest,
        encoded_image: &[u8],
    ) -> Result<PreparedVisionPrompt> {
        self.prepare_image_with_settings(request, encoded_image, self.generation)
    }

    pub(crate) fn prepare_image_with_settings(
        &self,
        request: &ChatCompletionRequest,
        encoded_image: &[u8],
        generation: GenerationSettings,
    ) -> Result<PreparedVisionPrompt> {
        self.prepare_image_with_limits(request, encoded_image, generation, None)
    }

    fn prepare_image_with_limits(
        &self,
        request: &ChatCompletionRequest,
        encoded_image: &[u8],
        generation: GenerationSettings,
        limits: Option<VisionLimits>,
    ) -> Result<PreparedVisionPrompt> {
        let readiness = self.vision_readiness.as_ref().ok_or_else(|| {
            models::ModelsError::InvalidConfig(
                "loaded model has no supported image execution contract".into(),
            )
        })?;
        if !readiness.is_ready() {
            return Err(models::ModelsError::InvalidConfig(format!(
                "image execution contract is incomplete: {} required tensors are missing",
                readiness.missing.len()
            ))
            .into());
        }
        match (self.vision.as_ref(), self.image_processor.as_ref()) {
            (
                Some(VisionConfig::PooledEncoder(vision)),
                Some(ImageProcessorConfig::Pooled(processor)),
            ) => {
                let request = self.with_image_token(request, vision.image_token_id)?;
                let prepared = self.prepare_with_settings(&request, generation)?;
                let image = match limits {
                    Some(limits) => processor.preprocess_encoded_with_patch_limit(
                        encoded_image,
                        limits.pooled_patch_limit(vision),
                    )?,
                    None => processor.preprocess_encoded(encoded_image)?,
                };
                if let Some(limits) = limits {
                    limits.validate(
                        image.grid_height.saturating_mul(image.grid_width),
                        vision.num_attention_heads,
                    )?;
                }
                let tokens =
                    PooledPromptTokens::prepare(&prepared.tokens.token_ids, &image, vision)?;
                validate_context(
                    tokens.token_ids.len(),
                    generation.max_tokens,
                    self.metadata.context_len,
                )?;
                Ok(PreparedVisionPrompt::Pooled { prompt: prepared.prompt, tokens, image })
            },
            (
                Some(VisionConfig::SpatialMergeEncoder(vision)),
                Some(ImageProcessorConfig::SpatialMerge(processor)),
            ) => {
                let request = self.with_image_token(request, vision.image_token_id)?;
                let prepared = self.prepare_with_settings(&request, generation)?;
                let image = match limits {
                    Some(limits) => processor.preprocess_encoded_with_max_pixels(
                        encoded_image,
                        limits.spatial_pixel_limit(vision),
                    )?,
                    None => processor.preprocess_encoded(encoded_image)?,
                };
                if let Some(limits) = limits {
                    limits.validate(
                        image
                            .grid_t
                            .saturating_mul(image.grid_height)
                            .saturating_mul(image.grid_width),
                        vision.num_attention_heads,
                    )?;
                }
                let tokens =
                    SpatialMergePromptTokens::prepare(&prepared.tokens.token_ids, &image, vision)?;
                validate_context(
                    tokens.token_ids.len(),
                    generation.max_tokens,
                    self.metadata.context_len,
                )?;
                Ok(PreparedVisionPrompt::SpatialMerge { prompt: prepared.prompt, tokens, image })
            },
            _ => Err(models::ModelsError::InvalidConfig(
                "loaded model has no complete supported image execution contract".into(),
            )
            .into()),
        }
    }

    fn with_image_token(
        &self,
        request: &ChatCompletionRequest,
        image_token_id: u32,
    ) -> Result<ChatCompletionRequest> {
        let occurrences = request
            .messages
            .iter()
            .map(|message| message.content.matches(IMAGE_PLACEHOLDER).count())
            .sum::<usize>();
        if occurrences == 0 {
            return Ok(request.clone());
        }
        if occurrences != 1 {
            return Err(models::ModelsError::InvalidConfig(
                "vision requests require exactly one image placeholder".into(),
            )
            .into());
        }
        let token = self.tokenizer.token(image_token_id).ok_or_else(|| {
            models::ModelsError::InvalidConfig(format!(
                "image token id {image_token_id} is missing from the tokenizer"
            ))
        })?;
        let mut request = request.clone();
        for message in &mut request.messages {
            message.content = message.content.replace(IMAGE_PLACEHOLDER, &token);
        }
        Ok(request)
    }
}

impl Model {
    /// Renders, tokenizes, and preprocesses one encoded image according to the
    /// contract discovered from the loaded checkpoint.
    pub fn prepare_image(
        &self,
        request: &ChatCompletionRequest,
        encoded_image: &[u8],
    ) -> Result<PreparedVisionPrompt> {
        self.prepare_image_with_settings(request, encoded_image, self.inner.descriptor.generation)
    }

    pub(crate) fn prepare_image_with_settings(
        &self,
        request: &ChatCompletionRequest,
        encoded_image: &[u8],
        generation: GenerationSettings,
    ) -> Result<PreparedVisionPrompt> {
        self.inner.descriptor.prepare_image_with_limits(
            request,
            encoded_image,
            generation,
            Some(self.vision_limits()),
        )
    }
}
