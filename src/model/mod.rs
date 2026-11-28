mod admission;
mod automatic_cache;
mod cache;
pub mod constraints;
mod decode;
mod descriptor;
mod helpers;
mod kv_sizing;
mod library;
mod lifecycle;
mod memory;
mod prefill;
mod prompt;
mod remote;
mod vision;
mod warmup;

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

pub use admission::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, ArchitectureOutcome,
    BackendAdmissionReport, CheckSubject, CheckpointEncoding, EncodingKind,
    MODEL_FORMAT_REGISTRY_SCHEMA_VERSION, WeightEncoding,
};
use architecture::InspectedModel;
pub use cache::CacheStatsReader;
use foundation::{
    conversation::Conversation,
    model::{BackendTarget, ModelManifest},
};
pub use memory::UnloadedTensors;
use models::{
    chat::ChatTemplate,
    execution::{TaskExecutionPlan, TaskKind},
    generation::{GenerationConfig, GenerationOverrides, GenerationSettings},
    layout::WeightContainer,
    tokenizer::{TextTokenizer, TokenizerValidation},
};
pub use remote::{RemoteModelContract, RemoteTaskMetadata, RemoteVisionContract};
use runtime::backend::ModelHandle;
pub use vision::{IMAGE_PLACEHOLDER, PreparedVisionPrompt};

use self::helpers::{model_id, validate_context};
pub use self::{
    cache::cohort::FillClaim,
    descriptor::{PreparedPrompt, PromptPreparationTimings},
    library::{LoadPreview, ModelLoadOptions},
};
use crate::{Engine, Error, Result, RuntimeConfig, Session, scheduler::ModelCoordinator};

/// Local checkpoint source accepted by [`ModelDescriptor::inspect_source`] and
/// [`Library::load_source`].
#[derive(Debug, Clone, Copy)]
pub enum ModelSource<'a> {
    /// Local Hugging Face-style checkpoint directory.
    Directory(&'a Path),
    /// Direct local GGUF text model (sibling shards are discovered
    /// automatically) with an optional `mmproj` vision projector.
    Gguf {
        /// Primary GGUF text shard path.
        text: &'a Path,
        /// Optional `mmproj` vision projector path.
        vision: Option<&'a Path>,
    },
}

/// Parsed model metadata and assets, ready for prompt preparation or backend
/// loading.
pub struct ModelDescriptor {
    inspected: InspectedModel,
    generation: GenerationSettings,
    unloaded: Option<UnloadedTensors>,
    template: ChatTemplate,
    tokenizer: TextTokenizer,
    tool_parser_factory:
        std::sync::OnceLock<std::result::Result<llguidance::ParserFactory, String>>,
    tokenizer_validation: TokenizerValidation,
}

/// Lazily initialized inference library configured for one accelerator backend.
#[derive(Debug, Clone)]
pub struct Library {
    state: Arc<Mutex<library::LibraryState>>,
    memory: memory::admission::ModelMemoryManager,
    caches: cache::KvCachePools,
    registry: kv_sizing::ModelRegistry,
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
    cache_cohort: cache::cohort::CacheCohort,
    coordinator: ModelCoordinator,
    kv_sizing: Mutex<kv_sizing::KvSizing>,
    registry: kv_sizing::ModelRegistry,
    memory: memory::admission::ModelMemoryLease,
}

impl ModelDescriptor {
    /// Inspects a local checkpoint directory or GGUF file without loading
    /// weights onto a device.
    pub fn inspect(path: impl AsRef<Path>, overrides: GenerationOverrides) -> Result<Self> {
        let path = path.as_ref();
        let source = if path.is_file() && WeightContainer::of(path) == Some(WeightContainer::Gguf) {
            ModelSource::Gguf { text: path, vision: None }
        } else {
            ModelSource::Directory(path)
        };
        Self::inspect_source(source, overrides)
    }

    /// Inspects a [`ModelSource`] (local checkpoint directory or direct GGUF
    /// text + optional `mmproj` files) without loading weights onto a device.
    pub fn inspect_source(source: ModelSource<'_>, overrides: GenerationOverrides) -> Result<Self> {
        let inspected = match source {
            ModelSource::Directory(path) => InspectedModel::inspect_directory(path)?,
            ModelSource::Gguf { text, vision } => InspectedModel::inspect_gguf(text, vision)?,
        };
        let generation_config = GenerationConfig::from_layout(&inspected.layout)?;
        let unloaded = memory::unloaded(
            &inspected.catalog,
            inspected.contract.as_ref().map(|execution| &execution.bindings),
            inspected.vision.as_ref(),
        );
        let tokenizer = TextTokenizer::from_layout(&inspected.layout)?;
        let tokenizer_validation = descriptor::validate_tokenizer(
            &inspected.task_plan,
            inspected.vision.as_ref(),
            &tokenizer,
        )?;
        let template = ChatTemplate::from_layout(&inspected.layout)?;
        Ok(Self {
            generation: generation_config.resolve(overrides)?,
            unloaded,
            template,
            tokenizer,
            tokenizer_validation,
            tool_parser_factory: std::sync::OnceLock::new(),
            inspected,
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
            id: foundation::model::ModelId::new(model_id(&self.inspected.layout.root)?),
            path: self.inspected.layout.root.display().to_string(),
            tokenizer_path: self
                .inspected
                .layout
                .primary_tokenizer_path()
                .map(|path| path.display().to_string()),
            context_len: self.inspected.metadata.context_len,
            quantization: self.inspected.metadata.quantization.clone(),
            preferred_backends,
        })
    }
}

fn task_mismatch(requested: TaskKind, task: &TaskExecutionPlan) -> Error {
    Error::TaskMismatch { requested, actual: task.task().kind() }
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

    /// Drafts each speculative decode step verifies, when the model drafts.
    pub(crate) fn draft_tokens(&self) -> Option<usize> {
        self.inner
            .config
            .scheduler
            .speculation
            .filter(|_| {
                self.inner.descriptor.inspected.contract.as_ref().is_some_and(|execution| {
                    !execution.semantic.decoder.prediction_layers.is_empty()
                })
            })
            .map(|speculation| {
                memory::prediction::backend_drafts(
                    self.engine().target(),
                    speculation.draft_tokens.get(),
                )
            })
    }

    /// Renders, tokenizes, and validates a chat request for this model.
    pub fn prepare(&self, conversation: &Conversation) -> Result<PreparedPrompt> {
        self.inner.descriptor.prepare(conversation)
    }
}

#[cfg(test)]
mod tests;
