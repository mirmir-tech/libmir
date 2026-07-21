use models::{
    chat::ChatTemplate,
    execution::{DecoderExecutionContract, ModelTask, TaskExecutionPlan},
    generation::{GenerationOverrides, GenerationSettings},
    layout::{DecoderConfig, ImageProcessorConfig, ModelLayout, ModelMetadata, VisionConfig},
    tokenizer::TextTokenizer,
    weights::TensorReadiness,
};

use super::ModelDescriptor;
use crate::Result;

impl ModelDescriptor {
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

    /// Resolves per-request overrides against the model generation
    /// configuration.
    pub fn resolve_generation(&self, overrides: GenerationOverrides) -> Result<GenerationSettings> {
        Ok(self.generation_config.resolve(overrides)?)
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
}
