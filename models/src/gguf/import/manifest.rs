//! `import.json`: what an import was made from and the files it holds. It is
//! written last, so a reader tells a complete, current import from file
//! names and metadata alone, without reading any GGUF header.

use std::{
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// The manifest's file name inside an import.
pub const MANIFEST: &str = "import.json";

/// The record of one published import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportManifest {
    /// The text model's shards, in order.
    pub text: Vec<SourceFile>,
    /// The vision tower's shards; empty for a text-only import.
    pub vision: Vec<SourceFile>,
    /// Every file the import holds, derived or linked, by name.
    pub files: Vec<String>,
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
    pub(super) fn of(path: &Path) -> Result<Self> {
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
        self.files.iter().all(|name| directory.join(name).exists())
            && self.sources().all(SourceFile::is_unchanged)
    }

    /// The text model's shards, then the vision tower's.
    pub fn sources(&self) -> impl Iterator<Item = &SourceFile> {
        self.text.iter().chain(&self.vision)
    }
}
