use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::error::{ModelsError, Result};

#[derive(Debug, Clone)]
pub struct WeightFile {
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub root: PathBuf,
    pub config_path: PathBuf,
    pub configuration_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub vocab_path: Option<PathBuf>,
    pub merges_path: Option<PathBuf>,
    pub added_tokens_path: Option<PathBuf>,
    pub special_tokens_map_path: Option<PathBuf>,
    pub tokenizer_config_path: Option<PathBuf>,
    pub modules_path: Option<PathBuf>,
    pub sentence_transformers_config_path: Option<PathBuf>,
    pub generation_config_path: Option<PathBuf>,
    pub model_spec_path: Option<PathBuf>,
    pub chat_template_path: Option<PathBuf>,
    pub kv_config_path: Option<PathBuf>,
    pub processor_config_path: Option<PathBuf>,
    pub preprocessor_config_path: Option<PathBuf>,
    pub video_processor_config_path: Option<PathBuf>,
    pub safetensors_index_path: Option<PathBuf>,
    pub weights: Vec<WeightFile>,
}

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: std::collections::BTreeMap<String, String>,
}

impl ModelLayout {
    pub fn inspect(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let config_path = required(root.join("config.json"))?;
        let configuration_path = optional(root.join("configuration.json"))
            .or_else(|| optional(root.join("params.json")));
        let vocab_path = optional(root.join("vocab.json"));
        let tokenizer_path = optional(root.join("tokenizer.json"))
            .or_else(|| optional(root.join("tokenizer.model")))
            .or_else(|| vocab_path.clone());
        let merges_path = optional(root.join("merges.txt"));
        let added_tokens_path = optional(root.join("added_tokens.json"));
        let special_tokens_map_path = optional(root.join("special_tokens_map.json"));
        let tokenizer_config_path = optional(root.join("tokenizer_config.json"));
        let modules_path = optional(root.join("modules.json"));
        let sentence_transformers_config_path =
            optional(root.join("config_sentence_transformers.json"));
        let generation_config_path = optional(root.join("generation_config.json"));
        let model_spec_path = optional(root.join("mir-model-spec.toml"));
        let chat_template_path = optional(root.join("chat_template.jinja"));
        let kv_config_path = optional(root.join("kv_config.json"));
        let processor_config_path = optional(root.join("processor_config.json"));
        let preprocessor_config_path = optional(root.join("preprocessor_config.json"));
        let video_processor_config_path = optional(root.join("video_processor_config.json"))
            .or_else(|| optional(root.join("video_preprocessor_config.json")));
        let safetensors_index_path = optional(root.join("model.safetensors.index.json"));
        let weights = read_weights(&root, safetensors_index_path.as_deref())?;

        Ok(Self {
            root,
            config_path,
            configuration_path,
            tokenizer_path,
            vocab_path,
            merges_path,
            added_tokens_path,
            special_tokens_map_path,
            tokenizer_config_path,
            modules_path,
            sentence_transformers_config_path,
            generation_config_path,
            model_spec_path,
            chat_template_path,
            kv_config_path,
            processor_config_path,
            preprocessor_config_path,
            video_processor_config_path,
            safetensors_index_path,
            weights,
        })
    }

    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer_path.is_some()
    }
}

fn required(path: PathBuf) -> Result<PathBuf> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(ModelsError::MissingFile(path))
    }
}

fn optional(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

fn read_weights(root: &Path, index_path: Option<&Path>) -> Result<Vec<WeightFile>> {
    let names = match index_path {
        Some(path) => read_indexed_names(path)?,
        None => read_safetensor_names(root)?,
    };
    names.into_iter().map(|name| weight_file(root.join(name))).collect()
}

fn read_indexed_names(path: &Path) -> Result<BTreeSet<String>> {
    let json = fs::read_to_string(path)?;
    let index: WeightIndex = serde_json::from_str(&json)?;
    Ok(index.weight_map.into_values().collect())
}

fn read_safetensor_names(root: &Path) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors")
            && let Some(name) = path.file_name().and_then(|name| name.to_str())
        {
            let _inserted = names.insert(name.to_owned());
        }
    }
    Ok(names)
}

fn weight_file(path: PathBuf) -> Result<WeightFile> {
    let Ok(metadata) = fs::metadata(&path) else {
        return Err(ModelsError::MissingFile(path));
    };
    Ok(WeightFile { path, bytes: metadata.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_mistral_params_as_supplemental_configuration() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("libmir-model-layout-params-{}", std::process::id()));
        fs::create_dir(&root)?;
        fs::write(root.join("config.json"), "{}")?;
        fs::write(root.join("params.json"), "{}")?;
        fs::write(root.join("model.safetensors"), [])?;

        let layout = ModelLayout::inspect(&root)?;
        fs::remove_dir_all(&root)?;

        assert_eq!(layout.configuration_path, Some(root.join("params.json")));
        Ok(())
    }
}
