mod cache;
mod descriptor;
mod helpers;
mod library;
mod lifecycle;
mod memory;
mod vision;

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use foundation::{
    model::{BackendTarget, ModelManifest},
    protocol::ChatCompletionRequest,
};
use models::{
    chat::{ChatPrompt, ChatTemplate},
    execution::{DecoderExecutionContract, ModelTask, TaskExecutionPlan},
    generation::{GenerationConfig, GenerationOverrides, GenerationSettings},
    layout::{DecoderConfig, ImageProcessorConfig, ModelLayout, ModelMetadata, VisionConfig},
    tokenizer::{TextTokenizer, TokenizedPrompt},
    weights::{TensorCatalog, TensorReadiness, VisionTensorSchema},
};
use runtime::{backend::ModelHandle, kv::KvCache};
pub use vision::{IMAGE_PLACEHOLDER, PreparedVisionPrompt};

use self::helpers::{model_id, validate_context};
use crate::{Engine, Error, Result, RuntimeConfig, Session, scheduler::DecodeCoordinator};

/// Parsed model metadata and assets, ready for prompt preparation or backend
/// loading.
pub struct ModelDescriptor {
    layout: ModelLayout,
    metadata: ModelMetadata,
    decoder: Option<DecoderConfig>,
    execution: Option<DecoderExecutionContract>,
    task_plan: TaskExecutionPlan,
    generation_config: GenerationConfig,
    generation: GenerationSettings,
    vision: Option<VisionConfig>,
    vision_readiness: Option<TensorReadiness>,
    image_processor: Option<ImageProcessorConfig>,
    template: ChatTemplate,
    tokenizer: TextTokenizer,
}

#[derive(Debug, Clone)]
/// Rendered prompt and tokenization produced before backend execution.
pub struct PreparedPrompt {
    /// Rendered chat prompt, including the selected template behavior.
    pub prompt: ChatPrompt,
    /// Tokenized prompt passed to the inference backend.
    pub tokens: TokenizedPrompt,
}

/// Lazily initialized inference library configured for one accelerator backend.
#[derive(Debug, Clone)]
pub struct Library {
    engine: Arc<Mutex<Option<Engine>>>,
    config: RuntimeConfig,
}

/// Loaded model that owns backend resources and creates independent sessions.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

struct ModelInner {
    descriptor: ModelDescriptor,
    engine: Engine,
    handle: ModelHandle,
    config: RuntimeConfig,
    cache: Mutex<KvCache>,
    coordinator: DecodeCoordinator,
}

impl ModelDescriptor {
    /// Inspects a Hugging Face-style model directory without loading weights
    /// onto a device.
    pub fn inspect(path: impl AsRef<Path>, overrides: GenerationOverrides) -> Result<Self> {
        let layout = ModelLayout::inspect(path)?;
        let metadata = ModelMetadata::from_layout(&layout)?;
        let generation_config = GenerationConfig::from_layout(&layout)?;
        let catalog = TensorCatalog::from_layout(&layout)?;
        let task_plan = TaskExecutionPlan::discover(&layout, &catalog)?;
        let decoder = match &task_plan {
            TaskExecutionPlan::Generation { decoder }
            | TaskExecutionPlan::Embedding { decoder, .. } => Some(decoder.clone()),
            TaskExecutionPlan::SequenceScoring { .. } => None,
        };
        let execution = decoder
            .as_ref()
            .map(|decoder| DecoderExecutionContract::discover(&layout, decoder, &catalog))
            .transpose()?;
        let vision = VisionConfig::from_layout(&layout)?;
        let vision_readiness = vision
            .as_ref()
            .map(|config| VisionTensorSchema::discover(config).readiness(&catalog));
        let image_processor = vision
            .as_ref()
            .map(|vision| ImageProcessorConfig::from_layout(&layout, vision.pipeline()))
            .transpose()?
            .flatten();
        Ok(Self {
            decoder,
            execution,
            task_plan,
            generation: generation_config.resolve(overrides)?,
            generation_config,
            vision,
            vision_readiness,
            image_processor,
            template: ChatTemplate::from_layout(&layout)?,
            tokenizer: TextTokenizer::from_layout(&layout)?,
            layout,
            metadata,
        })
    }

    /// Renders and tokenizes a request, validating it against the model context
    /// window.
    pub fn prepare(&self, request: &ChatCompletionRequest) -> Result<PreparedPrompt> {
        if !matches!(self.task_plan.task(), ModelTask::Generation) {
            return Err(task_mismatch("generation", &self.task_plan));
        }
        self.prepare_with_settings(request, self.generation)
    }

    pub(crate) fn prepare_with_settings(
        &self,
        request: &ChatCompletionRequest,
        generation: GenerationSettings,
    ) -> Result<PreparedPrompt> {
        let prompt = self.template.render(request)?;
        let tokens = self
            .tokenizer
            .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;
        if tokens.token_ids.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        validate_context(tokens.token_ids.len(), generation.max_tokens, self.metadata.context_len)?;
        Ok(PreparedPrompt { prompt, tokens })
    }

    /// Builds the backend-neutral manifest used to identify and load this
    /// model.
    pub fn manifest(&self) -> Result<ModelManifest> {
        self.manifest_with_backends(Vec::new())
    }

    pub(crate) fn manifest_for(&self, target: BackendTarget) -> Result<ModelManifest> {
        self.manifest_with_backends(vec![target])
    }

    fn manifest_with_backends(
        &self,
        preferred_backends: Vec<BackendTarget>,
    ) -> Result<ModelManifest> {
        Ok(ModelManifest {
            id: model_id(&self.layout.root)?,
            path: self.layout.root.display().to_string(),
            tokenizer_path: self
                .layout
                .tokenizer_path
                .as_ref()
                .map(|path| path.display().to_string()),
            context_len: self.metadata.context_len,
            quantization: self.metadata.quantization.clone(),
            preferred_backends,
        })
    }
}

fn task_mismatch(requested: &'static str, task: &TaskExecutionPlan) -> Error {
    let actual = match task.task() {
        ModelTask::Generation => "generation",
        ModelTask::Embedding(_) => "embedding",
        ModelTask::SequenceScoring(_) => "sequence scoring",
    };
    Error::TaskMismatch { requested, actual }
}

impl Model {
    #[must_use]
    /// Returns the inspected descriptor associated with this loaded model.
    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.inner.descriptor
    }

    #[must_use]
    /// Returns the accelerator engine owning this model.
    pub fn engine(&self) -> &Engine {
        &self.inner.engine
    }

    #[must_use]
    /// Returns the backend handle used for low-level inference calls.
    pub fn handle(&self) -> &ModelHandle {
        &self.inner.handle
    }

    #[must_use]
    /// Creates an independent generation session with its own K/V state.
    pub fn session(&self) -> Session {
        Session::new(self.clone(), self.inner.config.kv_cache.block_size)
    }

    /// Renders, tokenizes, and validates a chat request for this model.
    pub fn prepare(&self, request: &ChatCompletionRequest) -> Result<PreparedPrompt> {
        self.inner.descriptor.prepare(request)
    }

    pub(crate) fn decode_sequence(
        &self,
        sequence: runtime::backend::DecodeSequence,
    ) -> Result<runtime::backend::DecodeOutput> {
        self.inner.coordinator.submit(sequence)
    }
}

#[cfg(test)]
mod tests;
