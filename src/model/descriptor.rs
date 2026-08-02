use foundation::model::BackendTarget;
use models::{
    chat::ChatTemplate,
    execution::{DecoderExecutionContract, ModelTask, TaskExecutionPlan},
    generation::{GenerationOverrides, GenerationSettings},
    layout::{DecoderConfig, ImageProcessorConfig, ModelLayout, ModelMetadata, VisionConfig},
    tokenizer::{TextTokenizer, TokenizerValidation},
    weights::TensorReadiness,
};

use super::ModelDescriptor;
use crate::Result;

impl ModelDescriptor {
    #[must_use]
    /// Describes the physical checkpoint encodings discovered from tensor
    /// bindings.
    pub fn checkpoint_encoding(&self) -> super::CheckpointEncoding {
        let mut encoding = self.execution.as_ref().map_or_else(
            || match &self.task_plan {
                TaskExecutionPlan::SequenceScoring { bindings, .. } => {
                    super::CheckpointEncoding::from_encoder_bindings(bindings)
                },
                TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::Embedding { .. } => {
                    super::CheckpointEncoding::default()
                },
            },
            |execution| super::CheckpointEncoding::from_bindings(&execution.bindings),
        );
        if let Some(readiness) = &self.vision_readiness {
            encoding.include_dense_dtypes(&readiness.dtypes);
        }
        encoding
    }

    #[must_use]
    /// Reports conservative static compatibility with one accelerator backend.
    pub fn admission(&self, backend: BackendTarget) -> super::BackendAdmissionReport {
        super::admission::inspect(
            self.execution.as_ref(),
            &self.task_plan,
            self.vision.as_ref(),
            self.vision_readiness.as_ref(),
            self.image_processor.as_ref(),
            backend,
        )
    }

    #[must_use]
    /// Returns normalized task and decoder capabilities required by this
    /// model.
    pub fn architecture_requirements(&self) -> models::execution::ArchitectureRequirements {
        models::execution::ArchitectureRequirements::discover(
            &self.task_plan,
            self.execution.as_ref().map(|execution| &execution.semantic),
        )
    }

    #[must_use]
    /// Returns the discovered paths and files that make up the model.
    pub const fn layout(&self) -> &ModelLayout {
        &self.layout
    }

    #[must_use]
    /// Returns normalized checkpoint and context metadata.
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    #[must_use]
    /// Returns decoder dimensions and attention-layer configuration.
    pub const fn decoder(&self) -> Option<&DecoderConfig> {
        self.decoder.as_ref()
    }

    #[must_use]
    /// Returns the semantic decoder and its physical weight bindings.
    pub const fn execution(&self) -> Option<&DecoderExecutionContract> {
        self.execution.as_ref()
    }

    #[must_use]
    /// Returns the task and backbone contract discovered from checkpoint files.
    pub const fn task_plan(&self) -> &TaskExecutionPlan {
        &self.task_plan
    }

    #[must_use]
    /// Returns the primary task exposed by this checkpoint.
    pub fn task(&self) -> ModelTask {
        self.task_plan.task()
    }

    #[must_use]
    /// Returns the discovered vision execution contract, when present.
    pub const fn vision(&self) -> Option<&VisionConfig> {
        self.vision.as_ref()
    }

    #[must_use]
    /// Returns whether all tensors required by the discovered vision contract
    /// are present in the checkpoint catalog.
    pub const fn vision_readiness(&self) -> Option<&TensorReadiness> {
        self.vision_readiness.as_ref()
    }

    #[must_use]
    /// Returns the image processor configuration shipped with the checkpoint.
    pub const fn image_processor(&self) -> Option<&ImageProcessorConfig> {
        self.image_processor.as_ref()
    }

    #[must_use]
    /// Returns the generation settings resolved during inspection.
    pub const fn generation(&self) -> GenerationSettings {
        self.generation
    }

    /// Applies per-request overrides to the settings resolved when the model
    /// was loaded.
    pub fn resolve_generation(&self, overrides: GenerationOverrides) -> Result<GenerationSettings> {
        Ok(self.generation.with_overrides(overrides)?)
    }

    #[must_use]
    /// Returns the chat template discovered from checkpoint files.
    pub const fn template(&self) -> &ChatTemplate {
        &self.template
    }

    #[must_use]
    /// Returns the tokenizer loaded from the model directory.
    pub const fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    #[must_use]
    /// Returns the full-content vocabulary and token-ID validation report.
    pub const fn tokenizer_validation(&self) -> &TokenizerValidation {
        &self.tokenizer_validation
    }
}

pub(super) fn validate_tokenizer(
    task: &TaskExecutionPlan,
    vision: Option<&VisionConfig>,
    tokenizer: &TextTokenizer,
) -> Result<TokenizerValidation> {
    let vocabulary = match task {
        TaskExecutionPlan::Generation { decoder }
        | TaskExecutionPlan::Embedding { decoder, .. } => decoder.vocab_size,
        TaskExecutionPlan::SequenceScoring { encoder, .. } => encoder.vocab_size,
    };
    let required = match vision {
        Some(VisionConfig::PooledEncoder(config)) => {
            vec![config.image_token_id, config.image_begin_token_id, config.image_end_token_id]
        },
        Some(VisionConfig::SpatialMergeEncoder(config)) => {
            vec![config.image_token_id, config.vision_start_token_id, config.vision_end_token_id]
        },
        None => Vec::new(),
    };
    Ok(tokenizer.validate_contract(vocabulary, &required)?)
}
