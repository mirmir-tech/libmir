use models::generation::GenerationSettings;
use runtime::backend::{PrefillOutput, SamplingLogits};

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::PreparedVisionPrompt;
use crate::{Model, PreparedPrompt, ProgressEvent, PromptPreparationTimings, Result, Session};

pub(super) enum PreparedGeneration {
    Text(PreparedPrompt),
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Vision(PreparedVisionPrompt),
}

impl PreparedGeneration {
    pub(super) fn new(
        model: &Model,
        request: &super::GenerationRequest,
        settings: GenerationSettings,
        encoded_image: Option<&[u8]>,
    ) -> Result<Self> {
        let Some(encoded_image) = encoded_image else {
            return Ok(Self::Text(model.descriptor().prepare_with_reasoning(
                &request.conversation,
                settings,
                request.reasoning,
            )?));
        };
        if request.reasoning != crate::ReasoningMode::ModelDefault {
            return Err(models::ModelsError::InvalidConfig(
                "explicit reasoning mode is not supported for image requests".into(),
            )
            .into());
        }
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            Ok(Self::Vision(model.prepare_image_with_settings(
                &request.conversation,
                encoded_image,
                settings,
            )?))
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let _ = encoded_image;
            Err(models::ModelsError::InvalidConfig(
                "image generation is not enabled for this backend".into(),
            )
            .into())
        }
    }

    pub(super) fn token_ids(&self) -> &[u32] {
        match self {
            Self::Text(prepared) => &prepared.tokens.token_ids,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(PreparedVisionPrompt::Pooled { tokens, .. }) => &tokens.token_ids,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(PreparedVisionPrompt::SpatialMerge { tokens, .. }) => &tokens.token_ids,
        }
    }

    pub(super) fn prompt_text(&self) -> &str {
        match self {
            Self::Text(prepared) => &prepared.prompt.text,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(
                PreparedVisionPrompt::Pooled { prompt, .. }
                | PreparedVisionPrompt::SpatialMerge { prompt, .. },
            ) => &prompt.text,
        }
    }

    pub(super) fn preparation_timings(&self) -> PromptPreparationTimings {
        match self {
            Self::Text(prepared) => prepared.timings,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(_) => PromptPreparationTimings::default(),
        }
    }

    pub(super) fn prefill(
        &self,
        session: &mut Session,
        reserved_tokens: usize,
        sampling: SamplingLogits,
        cancellation: &crate::CancellationToken,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        match self {
            Self::Text(prepared) => session.prefill_generation_reserved(
                &prepared.tokens.token_ids,
                &prepared.cache_checkpoints,
                reserved_tokens,
                sampling,
                cancellation,
                progress,
            ),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(prepared) => {
                session.prefill_vision_reserved(prepared, reserved_tokens, sampling, progress)
            },
        }
    }
}
