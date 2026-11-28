use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use serde_json::Value;

use super::{ModelLayout, WeightFile};
use crate::{
    error::Result,
    gguf::{GgufModel, Import},
};

/// In-memory checkpoint metadata and optional assembled vision `SafeTensors`
/// derived directly from GGUF headers without writing an import directory.
#[derive(Debug, Clone)]
pub struct DerivedAssets {
    pub config: Value,
    pub generation_config: Value,
    pub tokenizer: Value,
    pub tokenizer_config: Value,
    pub preprocessor_config: Option<Value>,
    pub assembled_safetensors: Option<(String, Arc<[u8]>)>,
}

impl ModelLayout {
    /// Inspects a local GGUF text model (discovering sibling shards
    /// automatically) and an optional `mmproj` vision projector directly in
    /// memory without creating an on-disk import directory.
    pub fn from_gguf(text: impl AsRef<Path>, vision: Option<&Path>) -> Result<Self> {
        let text_path = text.as_ref();
        let text_model = GgufModel::open(text_path)?;
        let mut import = Import::derive(&text_model)?;
        if let Some(vision_path) = vision {
            let mmproj = GgufModel::open(vision_path)?;
            import = import.with_vision(&mmproj, &text_model)?;
        }
        let mut weights = Vec::with_capacity(import.shards.len() + import.vision_shards.len());
        for shard in import.shards.iter().chain(&import.vision_shards) {
            weights.push(WeightFile::open(shard.clone())?);
        }
        let assembled_safetensors =
            import.assembled.map(|(name, bytes)| (name, Arc::<[u8]>::from(bytes)));
        let root = text_path.to_path_buf();
        Ok(Self {
            config_path: root.clone(),
            root,
            configuration_path: None,
            tokenizer_path: None,
            vocab_path: None,
            merges_path: None,
            added_tokens_path: None,
            special_tokens_map_path: None,
            tokenizer_config_path: None,
            modules_path: None,
            sentence_transformers_config_path: None,
            generation_config_path: None,
            model_spec_path: None,
            chat_template_path: None,
            kv_config_path: None,
            processor_config_path: None,
            preprocessor_config_path: None,
            video_processor_config_path: None,
            safetensors_index_path: None,
            weights,
            derived: Some(DerivedAssets {
                config: import.config,
                generation_config: import.generation_config,
                tokenizer: import.tokenizer,
                tokenizer_config: import.tokenizer_config,
                preprocessor_config: import.preprocessor_config,
                assembled_safetensors,
            }),
        })
    }

    pub fn read_config(&self) -> Result<Value> {
        if let Some(derived) = &self.derived {
            return Ok(derived.config.clone());
        }
        crate::layout::json::read(&self.config_path)
    }

    pub fn read_generation_config(&self) -> Result<Option<Value>> {
        if let Some(derived) = &self.derived {
            return Ok(Some(derived.generation_config.clone()));
        }
        self.generation_config_path
            .as_deref()
            .map(|path| Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?))
            .transpose()
    }

    pub fn read_tokenizer_config(&self) -> Result<Option<Value>> {
        if let Some(derived) = &self.derived {
            return Ok(Some(derived.tokenizer_config.clone()));
        }
        self.tokenizer_config_path
            .as_deref()
            .map(|path| Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?))
            .transpose()
    }

    pub fn read_preprocessor_config(&self) -> Result<Option<Value>> {
        if let Some(derived) = &self.derived {
            return Ok(derived.preprocessor_config.clone());
        }
        let Some(path) =
            self.processor_config_path.as_ref().or(self.preprocessor_config_path.as_ref())
        else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_str(&std::fs::read_to_string(path)?)?))
    }

    #[must_use]
    pub fn primary_tokenizer_path(&self) -> Option<PathBuf> {
        self.tokenizer_path
            .clone()
            .or_else(|| self.derived.as_ref().map(|_| self.root.clone()))
    }
}
