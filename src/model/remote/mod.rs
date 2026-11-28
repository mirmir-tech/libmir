mod vision;

use foundation::model::BackendTarget;
use models::{
    execution::{
        ArchitectureRequirements, CausalScoringTask, DecoderExecutionContract, EmbeddingTask,
        TaskExecutionPlan,
    },
    layout::{DecoderConfig, WeightContainer},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TextTensorLayout, WeightBindingPlan},
};
pub use vision::RemoteVisionContract;
use vision::remote_vision;

use super::{BackendAdmissionReport, CheckpointEncoding};
use crate::{Error, Result};

/// Optional task metadata fetched alongside remote model configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoteTaskMetadata<'a> {
    /// Sentence Transformers `modules.json` value.
    pub modules: Option<&'a serde_json::Value>,
    /// Configuration of the Pooling module referenced by `modules.json`.
    pub pooling: Option<&'a serde_json::Value>,
    /// Configuration of the `LogitScore` module referenced by `modules.json`.
    pub logit_score: Option<&'a serde_json::Value>,
    /// Optional `config_sentence_transformers.json` value.
    pub sentence_transformers: Option<&'a serde_json::Value>,
    /// Optional `processor_config.json` or `preprocessor_config.json` value.
    pub processor: Option<&'a serde_json::Value>,
}

/// Backend-neutral execution contract discovered from remote model metadata
/// and `SafeTensors` headers without downloading tensor payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteModelContract {
    execution: Option<DecoderExecutionContract>,
    task: TaskExecutionPlan,
    vision: Option<RemoteVisionContract>,
    container: Option<WeightContainer>,
}

impl RemoteModelContract {
    /// Discovers generation, Sentence Transformers embedding, or sequence
    /// scoring contracts using the same parsers as local model inspection.
    pub fn inspect(
        config: &serde_json::Value,
        catalog: &TensorCatalog,
        metadata: RemoteTaskMetadata<'_>,
    ) -> Result<Option<Self>> {
        let vision = remote_vision(config, catalog, metadata.processor)?;
        let container = container(catalog);
        if let (Some(modules), Some(pooling)) = (metadata.modules, metadata.pooling) {
            let task = EmbeddingTask::from_sentence_transformers_values(
                modules,
                pooling,
                metadata.sentence_transformers,
            )?;
            let tensors = TextTensorLayout::discover(catalog).ok_or_else(|| {
                Error::from(models::ModelsError::InvalidConfig(
                    "embedding text tensor namespace is incomplete".into(),
                ))
            })?;
            let decoder = DecoderConfig::from_value(config)?;
            let execution = decoder_execution(&decoder, config, catalog)?;
            return Ok(Some(Self {
                execution: Some(execution),
                task: TaskExecutionPlan::Embedding { decoder, task, tensors },
                vision,
                container,
            }));
        }
        if let Some(modules) = metadata.modules
            && CausalScoringTask::config_path(modules)?.is_some()
        {
            let config_value = metadata.logit_score.ok_or_else(|| {
                models::ModelsError::InvalidConfig("LogitScore configuration is missing".into())
            })?;
            let decoder = DecoderConfig::from_value(config)?;
            let task = CausalScoringTask::from_values(
                config_value,
                metadata.sentence_transformers,
                decoder.vocab_size,
            )?;
            return Ok(Some(Self {
                execution: Some(decoder_execution(&decoder, config, catalog)?),
                task: TaskExecutionPlan::CausalScoring { decoder, task },
                vision,
                container,
            }));
        }
        if let Some(task) = TaskExecutionPlan::discover_remote_sequence_scoring(config, catalog)? {
            return Ok(Some(Self { execution: None, task, vision, container }));
        }
        if generation_config(config) {
            return generation_contract(config, catalog, vision).map(Some);
        }
        Ok(None)
    }

    /// Builds a generation contract from a Hugging Face `config.json` value
    /// and a header-only tensor catalog.
    pub fn inspect_generation(config: &serde_json::Value, catalog: &TensorCatalog) -> Result<Self> {
        let vision = remote_vision(config, catalog, None)?;
        generation_contract(config, catalog, vision)
    }

    /// Builds a generation contract from the headers of a GGUF model's
    /// shards, derived as a local import derives its configuration.
    pub fn inspect_gguf(model: &models::gguf::GgufModel) -> Result<Self> {
        let config = models::gguf::Import::derive(model)?.config;
        generation_contract(&config, &model.mapped_catalog()?, None)
    }

    /// The container of the model's text weights, from the inspected
    /// headers' files.
    #[must_use]
    pub const fn weight_container(&self) -> Option<WeightContainer> {
        self.container
    }

    /// Reports physical and architectural compatibility for one backend.
    #[must_use]
    pub fn admission(&self, backend: BackendTarget) -> BackendAdmissionReport {
        super::admission::inspect(
            self.execution.as_ref(),
            &self.task,
            self.vision.as_ref().map(|vision| &vision.config),
            self.vision.as_ref().map(|vision| &vision.readiness),
            self.vision.as_ref().and_then(|vision| vision.processor.as_ref()),
            backend,
        )
    }

    /// Describes the physical encodings derived from remote tensor bindings.
    #[must_use]
    pub fn checkpoint_encoding(&self) -> CheckpointEncoding {
        let mut encoding = self.execution.as_ref().map_or_else(
            || match &self.task {
                TaskExecutionPlan::SequenceScoring { bindings, .. } => {
                    CheckpointEncoding::from_encoder_bindings(bindings)
                },
                TaskExecutionPlan::Generation { .. }
                | TaskExecutionPlan::CausalScoring { .. }
                | TaskExecutionPlan::Embedding { .. } => CheckpointEncoding::default(),
            },
            |execution| CheckpointEncoding::from_bindings(&execution.bindings),
        );
        if let Some(vision) = &self.vision {
            encoding.include_dense_dtypes(&vision.readiness.dtypes);
        }
        encoding
    }

    /// Returns normalized architecture capabilities required by the model.
    #[must_use]
    pub fn architecture_requirements(&self) -> ArchitectureRequirements {
        ArchitectureRequirements::discover(
            &self.task,
            self.execution.as_ref().map(|execution| &execution.semantic),
        )
    }

    /// Returns the semantic decoder and typed physical bindings when the task
    /// uses a decoder architecture.
    #[must_use]
    pub const fn execution(&self) -> Option<&DecoderExecutionContract> {
        self.execution.as_ref()
    }

    /// Returns the discovered remote task contract.
    #[must_use]
    pub const fn task(&self) -> &TaskExecutionPlan {
        &self.task
    }

    /// Returns the discovered remote vision contract, when present.
    #[must_use]
    pub const fn vision(&self) -> Option<&RemoteVisionContract> {
        self.vision.as_ref()
    }
}

fn generation_contract(
    config: &serde_json::Value,
    catalog: &TensorCatalog,
    vision: Option<RemoteVisionContract>,
) -> Result<RemoteModelContract> {
    let decoder = DecoderConfig::from_value(config)?;
    Ok(RemoteModelContract {
        execution: Some(decoder_execution(&decoder, config, catalog)?),
        task: TaskExecutionPlan::Generation { decoder },
        vision,
        container: container(catalog),
    })
}

fn container(catalog: &TensorCatalog) -> Option<WeightContainer> {
    WeightContainer::of_model(catalog.tensors.iter().map(|tensor| tensor.container))
}

/// The decoder's contract, its bindings discovered as `config` declares
/// the quantization, as a downloaded checkpoint's are.
fn decoder_execution(
    decoder: &DecoderConfig,
    config: &serde_json::Value,
    catalog: &TensorCatalog,
) -> Result<DecoderExecutionContract> {
    let semantic = SemanticModelSpec::discover(decoder, catalog)?;
    let bindings = WeightBindingPlan::discover_with_config(&semantic, catalog, config)?;
    Ok(DecoderExecutionContract { semantic, bindings })
}

fn generation_config(config: &serde_json::Value) -> bool {
    config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .any(|architecture| {
            architecture.contains("CausalLM") || architecture.contains("ConditionalGeneration")
        })
}

#[cfg(test)]
mod tests;
