mod admission;
mod automatic_cache;
mod cache;
mod cache_cohort;
mod decode;
mod descriptor;
mod helpers;
mod library;
mod lifecycle;
mod memory;
mod memory_admission;
mod memory_policy;
mod prefill;
mod remote;
mod vision;
mod warmup;

use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

pub use admission::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, BackendAdmissionReport,
    CheckpointEncoding, MODEL_FORMAT_REGISTRY_SCHEMA_VERSION, WeightEncoding,
};
use foundation::{
    conversation::Conversation,
    model::{BackendTarget, ModelManifest},
};
use models::{
    chat::ChatTemplate,
    execution::{DecoderExecutionContract, ModelTask, TaskExecutionPlan},
    generation::{GenerationConfig, GenerationOverrides, GenerationSettings},
    layout::{DecoderConfig, ImageProcessorConfig, ModelLayout, ModelMetadata, VisionConfig},
    tokenizer::{TextTokenizer, TokenizerValidation},
    weights::{TensorCatalog, TensorReadiness, VisionTensorSchema},
};
pub use remote::{RemoteModelContract, RemoteTaskMetadata, RemoteVisionContract};
use runtime::backend::ModelHandle;
pub use vision::{IMAGE_PLACEHOLDER, PreparedVisionPrompt};

use self::helpers::{model_id, validate_context};
pub use self::{
    descriptor::{PreparedPrompt, PromptPreparationTimings},
    library::ModelLoadOptions,
};
use crate::{Engine, Error, Result, RuntimeConfig, Session, scheduler::ModelCoordinator};

/// Parsed model metadata and assets, ready for prompt preparation or backend
/// loading.
pub struct ModelDescriptor {
    layout: ModelLayout,
    metadata: ModelMetadata,
    decoder: Option<DecoderConfig>,
    execution: Option<DecoderExecutionContract>,
    task_plan: TaskExecutionPlan,
    generation: GenerationSettings,
    vision: Option<VisionConfig>,
    vision_readiness: Option<TensorReadiness>,
    image_processor: Option<ImageProcessorConfig>,
    template: ChatTemplate,
    tokenizer: TextTokenizer,
    tokenizer_validation: TokenizerValidation,
}

/// Lazily initialized inference library configured for one accelerator backend.
#[derive(Debug, Clone)]
pub struct Library {
    state: Arc<Mutex<library::LibraryState>>,
    memory: memory_admission::ModelMemoryManager,
    caches: cache::KvCachePools,
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
    cache: Arc<cache::SharedKvCache>,
    cache_cohort: cache_cohort::CacheCohort,
    coordinator: ModelCoordinator,
    _memory: memory_admission::ModelMemoryLease,
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
        let tokenizer = TextTokenizer::from_layout(&layout)?;
        let tokenizer_validation =
            descriptor::validate_tokenizer(&task_plan, vision.as_ref(), &tokenizer)?;
        Ok(Self {
            decoder,
            execution,
            task_plan,
            generation: generation_config.resolve(overrides)?,
            vision,
            vision_readiness,
            image_processor,
            template: ChatTemplate::from_layout(&layout)?,
            tokenizer,
            tokenizer_validation,
            layout,
            metadata,
        })
    }

    /// Renders and tokenizes a request, validating it against the model context
    /// window.
    pub fn prepare(&self, conversation: &Conversation) -> Result<PreparedPrompt> {
        if !matches!(self.task_plan.task(), ModelTask::Generation) {
            return Err(task_mismatch("generation", &self.task_plan));
        }
        self.prepare_with_settings(conversation, self.generation)
    }

    pub(crate) fn prepare_with_settings(
        &self,
        conversation: &Conversation,
        generation: GenerationSettings,
    ) -> Result<PreparedPrompt> {
        let render_started = Instant::now();
        let prompt = self.template.render(conversation)?;
        let render = render_started.elapsed();
        let tokenize_started = Instant::now();
        let tokens = self
            .tokenizer
            .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;
        let tokenize = tokenize_started.elapsed();
        if tokens.token_ids.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        validate_context(tokens.token_ids.len(), generation.max_tokens, self.metadata.context_len)?;
        let cache_checkpoints = self.cache_checkpoints(conversation, &tokens.token_ids)?;
        Ok(PreparedPrompt {
            prompt,
            tokens,
            cache_checkpoints,
            timings: PromptPreparationTimings { render, tokenize },
        })
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
    pub fn prepare(&self, conversation: &Conversation) -> Result<PreparedPrompt> {
        self.inner.descriptor.prepare(conversation)
    }
}

#[cfg(test)]
mod tests;
