//! An import of a GGUF model into the files the rest of `libmir` reads:
//! `config.json`, `generation_config.json`, `tokenizer.json` and
//! `tokenizer_config.json` derived from the metadata, plus links to the GGUF
//! files, whose tensors the catalog reads in place.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{GgufModel, config, tokenizer};
use crate::error::{ModelsError, Result};

/// The derived files of one GGUF model, before they are written.
#[derive(Debug, Clone)]
pub struct Import {
    pub config: Value,
    pub generation_config: Value,
    pub tokenizer: Value,
    pub tokenizer_config: Value,
    pub shards: Vec<PathBuf>,
}

impl Import {
    pub fn derive(model: &GgufModel) -> Result<Self> {
        Ok(Self {
            config: config::config(model)?,
            generation_config: config::generation_config(model),
            tokenizer: tokenizer::tokenizer(model)?,
            tokenizer_config: tokenizer::tokenizer_config(model),
            shards: model.shards.iter().map(|(path, _)| path.clone()).collect(),
        })
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
