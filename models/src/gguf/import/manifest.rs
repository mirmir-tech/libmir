//! `import.json`: what an import was made from and the files it holds. It is
//! written last, so a reader tells a complete, current import from file
//! names and metadata alone, without reading any GGUF header.

use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};

use crate::{error::Result, layout::WeightContainer};

/// The manifest's file name inside an import.
pub const MANIFEST: &str = "import.json";

/// The record of one published import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportManifest {
    /// The text model's shards, in order.
    pub text: Vec<SourceFile>,
    /// The vision tower's shards; empty for a text-only import.
    pub vision: Vec<SourceFile>,
    /// Every file the import holds, by name, with how it came there.
    pub files: Vec<ImportedFile>,
}

/// One file of an import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedFile {
    pub name: String,
    pub origin: FileOrigin,
}

/// How a file came to be in an import, so a later import of the same
/// directory replaces or removes only what an import made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileOrigin {
    /// Written from the GGUF metadata.
    Derived,
    /// A link the import made to a source elsewhere.
    Linked,
    /// A source already in the directory under its name, which the import
    /// left as it was: never the import's to replace or remove.
    InPlace,
}

/// A source file as the import saw it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    /// Its canonical path.
    pub path: PathBuf,
    pub bytes: u64,
    /// Nanoseconds since the Unix epoch, when the file system reports them.
    pub modified: Option<u64>,
}

impl SourceFile {
    /// `path` as it is now.
    pub fn of(path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .and_then(|elapsed| u64::try_from(elapsed.as_nanos()).ok());
        Ok(Self {
            path: path.to_path_buf(),
            bytes: metadata.len(),
            modified,
        })
    }

    /// Whether `path` is a hard link to this source: the same size and time
    /// as the source has now.
    pub(super) fn is_linked_at(&self, path: &Path) -> bool {
        match (Self::of(&self.path), Self::of(path)) {
            (Ok(source), Ok(link)) => {
                source.modified.is_some()
                    && (source.bytes, source.modified) == (link.bytes, link.modified)
            },
            _ => false,
        }
    }

    /// Whether the file is still there with the same size and time.
    #[must_use]
    pub fn is_unchanged(&self) -> bool {
        Self::of(&self.path).is_ok_and(|now| now == *self)
    }
}

impl ImportManifest {
    /// The manifest of the import in `directory`; `None` when there is none
    /// or it cannot be read, as for an import an earlier version made.
    #[must_use]
    pub fn read(directory: &Path) -> Option<Self> {
        let bytes = fs::read(directory.join(MANIFEST)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Whether `directory` still holds every file of the import and each
    /// source is unchanged.
    #[must_use]
    pub fn is_current(&self, directory: &Path) -> bool {
        self.files.iter().all(|file| directory.join(&file.name).exists())
            && self.sources().all(SourceFile::is_unchanged)
    }

    /// The source the import linked under `name`, if it made that link.
    #[must_use]
    pub fn linked(&self, name: &OsStr) -> Option<&SourceFile> {
        self.files
            .iter()
            .any(|file| file.origin == FileOrigin::Linked && OsStr::new(&file.name) == name)
            .then(|| self.sources().find(|source| source.path.file_name() == Some(name)))
            .flatten()
    }

    /// The files of `directory` the import made, derived or linked.
    pub(super) fn made(&self) -> impl Iterator<Item = &ImportedFile> {
        self.files.iter().filter(|file| file.origin != FileOrigin::InPlace)
    }

    /// The GGUF files the import holds in `directory`, sorted: the model
    /// it was made of, whatever other GGUF files are beside it.
    #[must_use]
    pub fn gguf_files(&self, directory: &Path) -> Vec<PathBuf> {
        let mut files = self
            .files
            .iter()
            .map(|file| directory.join(&file.name))
            .filter(|path| WeightContainer::of(path) == Some(WeightContainer::Gguf))
            .collect::<Vec<_>>();
        files.sort();
        files
    }

    /// The text model's shards, then the vision tower's.
    pub fn sources(&self) -> impl Iterator<Item = &SourceFile> {
        self.text.iter().chain(&self.vision)
    }
}
