use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

mod version;

use super::WeightContainer;
use crate::{
    error::{ModelsError, Result},
    gguf::Listing,
};

#[derive(Debug, Clone)]
pub struct WeightFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub container: WeightContainer,
}

impl WeightFile {
    /// A weight file on disk; any other file is an error.
    pub fn open(path: PathBuf) -> Result<Self> {
        let Ok(metadata) = fs::metadata(&path) else {
            return Err(ModelsError::MissingFile(path));
        };
        let Some(container) = WeightContainer::of(&path) else {
            return Err(ModelsError::InvalidConfig(format!(
                "{} is not a SafeTensors or GGUF weight file",
                path.display()
            )));
        };
        Ok(Self { path, bytes: metadata.len(), container })
    }
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
    /// The model files in `root`, all of one version of it: a directory
    /// replaced while it is read, as a GGUF import is when republished, or
    /// whose entries change meanwhile, is read again, and one that keeps
    /// changing for about half a second is
    /// [`ModelsError::ChangedWhileRead`]. Where the system tells no
    /// directory's identity (off Unix) it is read once, unchecked. The
    /// layout names its files by path: opening them later may find a
    /// later version than the one inspected.
    pub fn inspect(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        version::one_version(
            root,
            || version::of(root),
            || Self::read(root.to_path_buf()),
            std::thread::sleep,
        )
    }

    fn read(root: PathBuf) -> Result<Self> {
        let listing = Listing::of(&root)?;
        let optional = |name: &str| optional(&listing, root.join(name));
        let config_path = required(&listing, root.join("config.json"))?;
        let configuration_path = optional("configuration.json").or_else(|| optional("params.json"));
        let vocab_path = optional("vocab.json");
        let tokenizer_path = optional("tokenizer.json")
            .or_else(|| optional("tokenizer.model"))
            .or_else(|| optional("tiktoken.model"))
            .or_else(|| vocab_path.clone());
        let merges_path = optional("merges.txt");
        let added_tokens_path = optional("added_tokens.json");
        let special_tokens_map_path = optional("special_tokens_map.json");
        let tokenizer_config_path = optional("tokenizer_config.json");
        let modules_path = optional("modules.json");
        let sentence_transformers_config_path = optional("config_sentence_transformers.json");
        let generation_config_path = optional("generation_config.json");
        let model_spec_path = optional("mir-model-spec.toml");
        let chat_template_path = optional("chat_template.jinja");
        let kv_config_path = optional("kv_config.json");
        let processor_config_path = optional("processor_config.json");
        let preprocessor_config_path = optional("preprocessor_config.json");
        let video_processor_config_path = optional("video_processor_config.json")
            .or_else(|| optional("video_preprocessor_config.json"));
        let safetensors_index_path = optional("model.safetensors.index.json");
        let weights = read_weights(&root, &listing, safetensors_index_path.as_deref())?;

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

    /// The container of the model's text weights; `None` without weight
    /// files. A GGUF import also holds a vision tower's `SafeTensors` file.
    #[must_use]
    pub fn container(&self) -> Option<WeightContainer> {
        WeightContainer::of_model(self.weights.iter().map(|weight| weight.container))
    }

    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer_path.is_some()
    }
}

fn required(listing: &Listing, path: PathBuf) -> Result<PathBuf> {
    if reads(listing, &path) && path.is_file() {
        Ok(path)
    } else {
        Err(ModelsError::MissingFile(path))
    }
}

fn optional(listing: &Listing, path: PathBuf) -> Option<PathBuf> {
    (reads(listing, &path) && path.is_file()).then_some(path)
}

/// Whether a loader reads `path`: in a GGUF import, of the files an import
/// writes only those it holds.
fn reads(listing: &Listing, path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| listing.reads(name))
}

fn read_weights(
    root: &Path,
    listing: &Listing,
    index_path: Option<&Path>,
) -> Result<Vec<WeightFile>> {
    let names = match index_path {
        Some(path) => read_indexed_names(path)?,
        None => read_safetensor_names(root, listing)?,
    };
    names.into_iter().map(|name| WeightFile::open(root.join(name))).collect()
}

fn read_indexed_names(path: &Path) -> Result<BTreeSet<String>> {
    let json = fs::read_to_string(path)?;
    let index: WeightIndex = serde_json::from_str(&json)?;
    Ok(index.weight_map.into_values().collect())
}

fn read_safetensor_names(root: &Path, listing: &Listing) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    // A GGUF import directory links its shards beside any assembled vision
    // `SafeTensors` file, and its manifest names both. An ordinary checkpoint
    // directory keeps one format: if non-projector `SafeTensors` weights exist
    // at its root, never merge `.gguf` files beside them.
    let safetensors = WeightContainer::SafeTensors.discover(root)?;
    let has_base_safetensors = safetensors.iter().any(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| !crate::gguf::is_projector_name(name))
    });
    let gguf = if listing.is_import() || !has_base_safetensors {
        crate::gguf::files_in(root)?
    } else {
        Vec::new()
    };
    for path in safetensors.into_iter().chain(gguf) {
        if let Some(name) = path.file_name().and_then(|name| name.to_str())
            && listing.reads(name)
        {
            let _inserted = names.insert(name.to_owned());
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests;
