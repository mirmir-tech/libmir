//! An import of a GGUF model into the files the rest of `libmir` reads:
//! `config.json`, `generation_config.json`, `tokenizer.json` and
//! `tokenizer_config.json` derived from the metadata, plus links to the GGUF
//! files, whose tensors the catalog reads in place.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{GgufModel, VisionImport, config, tokenizer};
use crate::error::{ModelsError, Result};

/// The derived files of one GGUF model, before they are written.
#[derive(Debug, Clone)]
pub struct Import {
    pub config: Value,
    pub generation_config: Value,
    pub tokenizer: Value,
    pub tokenizer_config: Value,
    pub shards: Vec<PathBuf>,
    /// `preprocessor_config.json` of a vision import.
    pub preprocessor_config: Option<Value>,
    /// `(file name, SafeTensors bytes)` the vision import assembles.
    pub assembled: Option<(String, Vec<u8>)>,
}

impl Import {
    pub fn derive(model: &GgufModel) -> Result<Self> {
        Ok(Self {
            config: config::config(model)?,
            generation_config: config::generation_config(model),
            tokenizer: tokenizer::tokenizer(model)?,
            tokenizer_config: tokenizer::tokenizer_config(model),
            shards: model.shards.iter().map(|(path, _)| path.clone()).collect(),
            preprocessor_config: None,
            assembled: None,
        })
    }

    /// Adds the vision tower of an `mmproj` file: the text config moves under
    /// `text_config`, as in the multimodal checkpoint.
    pub fn with_vision(mut self, mmproj: &GgufModel, text: &GgufModel) -> Result<Self> {
        let vision = VisionImport::derive(mmproj, text, &self.config)?;
        let mut root = vision.root;
        if let Some(tied) = self.config.get("tie_word_embeddings") {
            root.insert("tie_word_embeddings".into(), tied.clone());
        }
        // A vision import may adjust the text model (Gemma 4's bidirectional
        // image attention).
        root.entry("text_config").or_insert(self.config);
        self.config = Value::Object(root);
        self.preprocessor_config = Some(vision.preprocessor_config);
        self.assembled = vision.assembled;
        self.shards.extend(mmproj.shards.iter().map(|(path, _)| path.clone()));
        Ok(self)
    }

    /// Writes the import into `directory`, which then loads like a
    /// `SafeTensors` checkpoint directory. Shards are linked, not copied.
    pub fn write(&self, directory: &Path) -> Result<()> {
        std::fs::create_dir_all(directory)?;
        for (name, value) in [
            ("config.json", &self.config),
            ("generation_config.json", &self.generation_config),
            ("tokenizer.json", &self.tokenizer),
            ("tokenizer_config.json", &self.tokenizer_config),
        ] {
            std::fs::write(directory.join(name), serde_json::to_vec_pretty(value)?)?;
        }
        if let Some(value) = &self.preprocessor_config {
            std::fs::write(
                directory.join("preprocessor_config.json"),
                serde_json::to_vec_pretty(value)?,
            )?;
        }
        if let Some((name, bytes)) = &self.assembled {
            std::fs::write(directory.join(name), bytes)?;
        }
        for shard in &self.shards {
            let name = shard.file_name().ok_or_else(|| {
                ModelsError::InvalidGguf(format!("{} has no file name", shard.display()))
            })?;
            let link = directory.join(name);
            if link.symlink_metadata().is_ok() {
                std::fs::remove_file(&link)?;
            }
            link_file(&std::fs::canonicalize(shard)?, &link)?;
        }
        Ok(())
    }
}

#[cfg(unix)]
fn link_file(target: &Path, link: &Path) -> Result<()> {
    Ok(std::os::unix::fs::symlink(target, link)?)
}

#[cfg(not(unix))]
fn link_file(target: &Path, link: &Path) -> Result<()> {
    Ok(std::fs::hard_link(target, link)?)
}
