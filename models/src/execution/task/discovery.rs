use super::{ModelTask, PoolingMode, SequenceScoringTask, sentence_transformers};
use crate::{
    error::{ModelsError, Result},
    layout::{DecoderConfig, EncoderConfig, ModelLayout},
    weights::{EncoderTensorSchema, TensorCatalog, TextTensorLayout},
};

#[derive(Debug, Clone, PartialEq)]
pub enum TaskExecutionPlan {
    Generation {
        decoder: DecoderConfig,
    },
    Embedding {
        decoder: DecoderConfig,
        task: super::EmbeddingTask,
        tensors: TextTensorLayout,
    },
    SequenceScoring {
        encoder: EncoderConfig,
        task: SequenceScoringTask,
    },
}

impl TaskExecutionPlan {
    pub fn discover(layout: &ModelLayout, catalog: &TensorCatalog) -> Result<Self> {
        if let Some(task) = sentence_transformers::discover(layout)? {
            let tensors = TextTensorLayout::discover(catalog)
                .ok_or_else(|| invalid("embedding text tensor namespace is incomplete"))?;
            return Ok(Self::Embedding {
                decoder: DecoderConfig::from_layout(layout)?,
                task,
                tensors,
            });
        }
        if sequence_scoring_layout(catalog) {
            let encoder = EncoderConfig::from_layout(layout)?;
            if encoder.num_labels == 0 {
                return Err(invalid("sequence scoring requires at least one label"));
            }
            let readiness = EncoderTensorSchema::discover(&encoder, catalog).readiness(catalog);
            if !readiness.is_ready() {
                return Err(invalid(format!(
                    "sequence scoring tensor contract is incomplete: {} required tensors are missing",
                    readiness.missing.len()
                )));
            }
            return Ok(Self::SequenceScoring {
                task: SequenceScoringTask {
                    labels: encoder.num_labels,
                    pooling: PoolingMode::Cls,
                },
                encoder,
            });
        }
        Ok(Self::Generation {
            decoder: DecoderConfig::from_layout(layout)?,
        })
    }

    #[must_use]
    pub fn task(&self) -> ModelTask {
        match self {
            Self::Generation { .. } => ModelTask::Generation,
            Self::Embedding { task, .. } => ModelTask::Embedding(task.clone()),
            Self::SequenceScoring { task, .. } => ModelTask::SequenceScoring(*task),
        }
    }
}

fn sequence_scoring_layout(catalog: &TensorCatalog) -> bool {
    catalog.contains("new.embeddings.word_embeddings.weight")
        && catalog.contains("new.encoder.layer.0.attention.qkv_proj.weight")
        && catalog.contains("new.pooler.dense.weight")
        && catalog.contains("classifier.weight")
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
