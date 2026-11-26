//! An import of a GGUF model into the files the rest of `libmir` reads:
//! `config.json`, `generation_config.json`, `tokenizer.json` and
//! `tokenizer_config.json` derived from the metadata, plus links to the GGUF
//! files, whose tensors the catalog reads in place.

mod files;
mod manifest;
mod publish;

use std::path::{Path, PathBuf};

pub use manifest::{
    FileOrigin, ImportManifest, ImportedFile, Listing, MANIFEST, SourceFile, Stamp, Written,
};
use serde_json::Value;

use super::{GgufModel, VisionImport, config, tokenizer};
use crate::error::Result;

/// The derived files of one GGUF model, before they are written.
#[derive(Debug, Clone)]
pub struct Import {
    pub config: Value,
    pub generation_config: Value,
    pub tokenizer: Value,
    pub tokenizer_config: Value,
    /// The text model's shards, linked under their names.
    pub shards: Vec<PathBuf>,
    /// The vision tower's `mmproj` shards, linked like the text model's.
    pub vision_shards: Vec<PathBuf>,
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
            vision_shards: Vec::new(),
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
        self.vision_shards.extend(mmproj.shards.iter().map(|(path, _)| path.clone()));
        Ok(self)
    }

    /// Writes the import into `directory`, which then loads like a
    /// `SafeTensors` checkpoint directory. Shards are linked, not copied; a
    /// shard already in `directory`, such as when it is the source's own
    /// directory, stays as it is, as does every file no import made. Every
    /// path is resolved and checked before the first write; the manifest
    /// naming the new files is written aside before any file changes and
    /// moved into place last, and loaders refuse the directory meanwhile.
    /// Writers and publishers of one directory take turns.
    pub fn write(&self, directory: &Path) -> Result<()> {
        let _turn = publish::lock(directory)?;
        self.write_unlocked(directory)
    }

    fn write_unlocked(&self, directory: &Path) -> Result<()> {
        let plan = files::Plan::new(self, directory)?;
        std::fs::create_dir_all(directory)?;
        plan.write(directory)
    }

    /// Publishes the import as `directory` in one step: it is written beside
    /// it and replaces the previous import only once complete, so a failure
    /// leaves the last good import in place. No source may live inside
    /// `directory`, which is replaced whole; a symlink there is replaced as
    /// the link, never the directory it links to.
    pub fn publish(&self, directory: &Path) -> Result<()> {
        let sources = self.shards.iter().chain(&self.vision_shards).cloned().collect::<Vec<_>>();
        publish::replace(directory, &sources, |staging| self.write_unlocked(staging))
    }

    /// The derived files, `(name, contents)`, with `config.json` last.
    fn derived(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut files = Vec::new();
        for (name, value) in [
            ("generation_config.json", Some(&self.generation_config)),
            ("tokenizer.json", Some(&self.tokenizer)),
            ("tokenizer_config.json", Some(&self.tokenizer_config)),
            ("preprocessor_config.json", self.preprocessor_config.as_ref()),
        ] {
            if let Some(value) = value {
                files.push((name.to_owned(), serde_json::to_vec_pretty(value)?));
            }
        }
        if let Some((name, bytes)) = &self.assembled {
            files.push((name.clone(), bytes.clone()));
        }
        files.push(("config.json".to_owned(), serde_json::to_vec_pretty(&self.config)?));
        Ok(files)
    }
}

/// Why an import cannot be written where it was asked to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ImportConflict {
    /// A file the import did not make, or cannot tell it made, as where
    /// its manifest is gone or an earlier version did not record it, is
    /// kept: the error names it, and removing it lets the import be
    /// written there.
    #[error(
        "the file there has a shard's name, and the import did not make it or cannot tell it \
         did; remove it to write the import there"
    )]
    OccupiedLink,
    /// The same for a file the import derives from the GGUF metadata, such
    /// as a user's own `tokenizer.json` or one changed since it was written.
    #[error(
        "the file there has a derived file's name, and the import did not write it or cannot \
         tell it wrote it as it is now; remove it to write the import there"
    )]
    OccupiedFile,
    #[error("two files of the import share a name")]
    DuplicateName,
    #[error("a source file lives inside the directory publishing replaces")]
    SourceInside,
}

#[cfg(test)]
mod tests;
