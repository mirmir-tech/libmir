//! Checkpoint import defaults, not backend selection. Explicit configuration
//! and the complete mir-model-spec.toml take precedence over these values.
use serde_json::Value;

use crate::{error::Result, semantic::QkNormalization};

#[derive(Default)]
pub(super) struct Defaults {
    pub attention_scale: Option<f64>,
    pub qk_normalization: Option<QkNormalization>,
    embedding: Embedding,
}

#[derive(Default)]
enum Embedding {
    #[default]
    Unspecified,
    SqrtHiddenSize,
}

impl Defaults {
    pub fn normalization(&self, config: &Value) -> Result<Option<QkNormalization>> {
        Ok(config
            .get("qk_normalization")
            .filter(|value| !value.is_null())
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()?
            .or(self.qk_normalization))
    }

    pub fn embedding_multiplier(&self, hidden_size: usize) -> Result<Option<f64>> {
        Ok(match self.embedding {
            Embedding::Unspecified => None,
            Embedding::SqrtHiddenSize => Some(f64::from(u32::try_from(hidden_size)?).sqrt()),
        })
    }
}

pub(super) fn for_checkpoint(config: &Value) -> Defaults {
    // Gemma 4: established by our Metal 26B reference runs and the CUDA
    // 12B BF16 regression. See docs/semantic-model-migration.md for provenance.
    match config.get("model_type").and_then(Value::as_str) {
        Some("gemma4" | "gemma4_text" | "gemma4_unified") => Defaults {
            attention_scale: Some(1.0),
            qk_normalization: Some(QkNormalization::QueryKeyValueRms),
            embedding: Embedding::SqrtHiddenSize,
        },
        _ => Defaults::default(),
    }
}
