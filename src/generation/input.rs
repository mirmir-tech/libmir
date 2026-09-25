use models::generation::GenerationSettings;
use runtime::backend::{PrefillOutput, SamplingLogits};

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::PreparedVisionPrompt;
use crate::{Model, PreparedPrompt, ProgressEvent, PromptPreparationTimings, Result, Session};

enum Prepared {
    Text(PreparedPrompt),
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Vision(PreparedVisionPrompt),
}

/// Prepared prompt, counted as in preparation until its prefill returns.
pub(super) struct PreparedGeneration {
    prompt: Prepared,
    preparation: std::cell::Cell<Option<crate::scheduler::PreparationGuard>>,
}

impl PreparedGeneration {
    pub(super) fn normalizer(
        &self,
        tokenizer: &models::tokenizer::TextTokenizer,
        request: &super::GenerationRequest,
    ) -> models::generation::OutputNormalizer {
        let normalizer = self.tool_prefix().map_or_else(
            || models::generation::OutputNormalizer::new(tokenizer, self.prompt_text()),
            |prefix| {
                models::generation::OutputNormalizer::with_tool_prefix(
                    tokenizer,
                    self.prompt_text(),
                    prefix,
                )
            },
        );
        if request.tool_constraints == super::ToolConstraints::Schema {
            return normalizer.with_constrained_tool();
        }
        let conversation = &request.conversation;
        if conversation.tools.is_empty()
            || matches!(conversation.tool_choice, foundation::conversation::ToolChoice::None)
        {
            normalizer.without_tool_protocol()
        } else {
            normalizer
        }
    }

    pub(super) fn initial_tool_calls(&self) -> String {
        self.tool_prefix().map_or_else(String::new, models::chat::ToolCallPrefix::text)
    }

    pub(super) fn tool_prefix(&self) -> Option<&models::chat::ToolCallPrefix> {
        match &self.prompt {
            Prepared::Text(prepared) => prepared.prompt.tool_prefix.as_ref(),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Prepared::Vision(_) => None,
        }
    }

    pub(super) fn new(
        model: &Model,
        request: &super::GenerationRequest,
        settings: GenerationSettings,
        encoded_image: Option<&[u8]>,
        metrics: &mut runtime::metrics::GenerationMetricsRecorder,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        let preparation = std::cell::Cell::new(model.announce_preparation());
        let prompt = Prepared::new(model, request, settings, encoded_image)?;
        metrics.record_prompt(started.elapsed(), prompt.token_ids().len());
        let stages = prompt.preparation_timings();
        metrics.record_prompt_stages(stages.render, stages.tokenize);
        Ok(Self { prompt, preparation })
    }

    pub(super) fn token_ids(&self) -> &[u32] {
        self.prompt.token_ids()
    }

    pub(super) fn prompt_text(&self) -> &str {
        self.prompt.prompt_text()
    }

    pub(super) fn prefill(
        &self,
        session: &mut Session,
        reserved_tokens: usize,
        sampling: SamplingLogits,
        cancellation: &crate::CancellationToken,
        progress: &mut dyn FnMut(ProgressEvent),
        metrics: &mut runtime::metrics::GenerationMetricsRecorder,
    ) -> Result<PrefillOutput> {
        let started = std::time::Instant::now();
        let output =
            self.prompt.prefill(session, reserved_tokens, sampling, cancellation, progress);
        drop(self.preparation.take());
        let output = output?;
        cancellation.check()?;
        super::telemetry::record_prefill_metrics(metrics, started, self.token_ids().len(), &output);
        Ok(output)
    }
}

impl Prepared {
    fn new(
        model: &Model,
        request: &super::GenerationRequest,
        settings: GenerationSettings,
        encoded_image: Option<&[u8]>,
    ) -> Result<Self> {
        let Some(encoded_image) = encoded_image else {
            let conversation = super::constraints::prompt::conversation(
                &request.conversation,
                request.tool_constraints,
            );
            return Ok(Self::Text(model.descriptor().prepare_with_reasoning(
                &conversation,
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

    fn token_ids(&self) -> &[u32] {
        match self {
            Self::Text(prepared) => &prepared.tokens.token_ids,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(PreparedVisionPrompt::Pooled { tokens, .. }) => &tokens.token_ids,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(PreparedVisionPrompt::SpatialMerge { tokens, .. }) => &tokens.token_ids,
        }
    }

    fn prompt_text(&self) -> &str {
        match self {
            Self::Text(prepared) => &prepared.prompt.text,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(
                PreparedVisionPrompt::Pooled { prompt, .. }
                | PreparedVisionPrompt::SpatialMerge { prompt, .. },
            ) => &prompt.text,
        }
    }

    fn preparation_timings(&self) -> PromptPreparationTimings {
        match self {
            Self::Text(prepared) => prepared.timings,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Vision(_) => PromptPreparationTimings::default(),
        }
    }

    fn prefill(
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
