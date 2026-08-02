use foundation::model::BackendTarget;
use models::{
    execution::{
        ArchitectureRequirements, DecoderExecutionContract, EmbeddingTask, TaskExecutionPlan,
    },
    layout::{DecoderConfig, ImageProcessorConfig, VisionConfig},
    semantic::SemanticModelSpec,
    weights::{
        TensorCatalog, TensorReadiness, TextTensorLayout, VisionTensorSchema, WeightBindingPlan,
    },
};

use super::{BackendAdmissionReport, CheckpointEncoding};
use crate::{Error, Result};

/// Optional task metadata fetched alongside remote model configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoteTaskMetadata<'a> {
    /// Sentence Transformers `modules.json` value.
    pub modules: Option<&'a serde_json::Value>,
    /// Configuration of the Pooling module referenced by `modules.json`.
    pub pooling: Option<&'a serde_json::Value>,
    /// Optional `config_sentence_transformers.json` value.
    pub sentence_transformers: Option<&'a serde_json::Value>,
    /// Optional `processor_config.json` or `preprocessor_config.json` value.
    pub processor: Option<&'a serde_json::Value>,
}

/// Vision configuration, processor, and tensor readiness discovered from a
/// remote checkpoint without fetching tensor payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteVisionContract {
    config: VisionConfig,
    readiness: TensorReadiness,
    processor: Option<ImageProcessorConfig>,
}

/// Backend-neutral execution contract discovered from remote model metadata
/// and `SafeTensors` headers without downloading tensor payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteModelContract {
    execution: Option<DecoderExecutionContract>,
    task: TaskExecutionPlan,
    vision: Option<RemoteVisionContract>,
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
            let execution = decoder_execution(&decoder, catalog)?;
            return Ok(Some(Self {
                execution: Some(execution),
                task: TaskExecutionPlan::Embedding { decoder, task, tensors },
                vision,
            }));
        }
        if let Some(task) = TaskExecutionPlan::discover_remote_sequence_scoring(config, catalog)? {
            return Ok(Some(Self { execution: None, task, vision }));
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
                TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::Embedding { .. } => {
                    CheckpointEncoding::default()
                },
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

impl RemoteVisionContract {
    /// Returns the normalized vision execution configuration.
    #[must_use]
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Returns vision tensor readiness derived from `SafeTensors` headers.
    #[must_use]
    pub const fn readiness(&self) -> &TensorReadiness {
        &self.readiness
    }

    /// Returns the parsed image processor, when the repository provides one.
    #[must_use]
    pub const fn processor(&self) -> Option<&ImageProcessorConfig> {
        self.processor.as_ref()
    }
}

fn generation_contract(
    config: &serde_json::Value,
    catalog: &TensorCatalog,
    vision: Option<RemoteVisionContract>,
) -> Result<RemoteModelContract> {
    let decoder = DecoderConfig::from_value(config)?;
    Ok(RemoteModelContract {
        execution: Some(decoder_execution(&decoder, catalog)?),
        task: TaskExecutionPlan::Generation { decoder },
        vision,
    })
}

fn remote_vision(
    config: &serde_json::Value,
    catalog: &TensorCatalog,
    processor: Option<&serde_json::Value>,
) -> Result<Option<RemoteVisionContract>> {
    let declared = config.get("vision_config").is_some_and(serde_json::Value::is_object);
    let Some(config) = VisionConfig::from_value(config)? else {
        if declared {
            return Err(Error::from(models::ModelsError::InvalidConfig(
                "unsupported vision execution contract".into(),
            )));
        }
        return Ok(None);
    };
    let readiness = VisionTensorSchema::discover(&config).readiness(catalog);
    let processor = processor
        .map(|value| ImageProcessorConfig::from_value(value, config.pipeline()))
        .transpose()?;
    Ok(Some(RemoteVisionContract { config, readiness, processor }))
}

fn decoder_execution(
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
) -> Result<DecoderExecutionContract> {
    let semantic = SemanticModelSpec::discover(decoder, catalog)?;
    let bindings = WeightBindingPlan::discover(&semantic, catalog)?;
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
