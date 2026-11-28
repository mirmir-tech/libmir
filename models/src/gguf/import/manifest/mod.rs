//! `import.json`: what an import was made from and the files it holds. It is
//! written before any file it names is changed, beside its place, and moved
//! there last: a reader tells a complete, current import from file names
//! and metadata alone, without reading any GGUF header, and a write cut
//! short still tells which files it made.

mod intent;
mod listing;
mod source;
mod wire;

use std::{
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
};

pub(in crate::gguf::import) use intent::interrupted;
pub use listing::Listing;
use serde::{Deserialize, Serialize};
pub use source::{SourceFile, Stamp, Written};

use crate::{
    error::{ModelsError, Result},
    layout::WeightContainer,
};

/// The manifest's file name inside an import.
pub const MANIFEST: &str = "import.json";

/// The record of one published import, serialized with its format (see
/// `wire`).
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileOrigin {
    /// Written from the GGUF metadata, as `written`: replaced or removed
    /// only while it still holds those contents.
    Derived { written: Written },
    /// A link the import made itself, where nothing or its own earlier link
    /// was, to the source at `source`, one the manifest records, whatever
    /// name either has: the file `named`, the path the import was given,
    /// resolved to when it was made. A `named` link since turned to another
    /// file makes the import stale.
    Linked { source: PathBuf, named: PathBuf },
    /// A file the import found under its name and left as it was: a source
    /// at its own path, or a user's link to one, never the import's to
    /// replace or remove. It resolved to the recorded `source`, and must
    /// still resolve to it for the import to be current.
    InPlace { source: PathBuf },
}

impl ImportManifest {
    /// The manifest of the import in `directory`; `None` when there is none
    /// or it cannot be used, as for one naming a file outside `directory`.
    #[must_use]
    pub fn read(directory: &Path) -> Option<Self> {
        Self::load(directory).ok().flatten()
    }

    /// The manifest of the import in `directory`, `None` when there is none
    /// or it cannot be parsed, as for a later format. A manifest naming
    /// anything but a file directly in `directory` is refused, before any of
    /// its names is used.
    pub fn load(directory: &Path) -> Result<Option<Self>> {
        Self::load_at(&directory.join(MANIFEST))
    }

    pub(super) fn load_at(path: &Path) -> Result<Option<Self>> {
        let Ok(bytes) = fs::read(path) else {
            return Ok(None);
        };
        let Ok(manifest) = serde_json::from_slice::<Self>(&bytes) else {
            return Ok(None);
        };
        Self::check(&manifest, path)?;
        Ok(Some(manifest))
    }

    /// Refuses a manifest whose file names are not single file names, or
    /// that ties a file to a source it does not record.
    pub(super) fn check(&self, path: &Path) -> Result<()> {
        let recorded = |source: &PathBuf| self.sources().any(|recorded| recorded.path == *source);
        let invalid = self.files.iter().find(|file| {
            !is_file_name(&file.name)
                || match &file.origin {
                    FileOrigin::Derived { .. } => false,
                    FileOrigin::Linked { source, .. } | FileOrigin::InPlace { source } => {
                        !recorded(source)
                    },
                }
        });
        invalid.map_or(Ok(()), |file| {
            Err(ModelsError::InvalidImportEntry {
                manifest: path.to_path_buf(),
                name: file.name.clone(),
            })
        })
    }

    /// Whether `directory` still holds every file of the import as the
    /// import left it, each derived one as it wrote it, each linked or found
    /// in place still resolving to its own source, each source is unchanged,
    /// and no later write of it was cut short.
    #[must_use]
    pub fn is_current(&self, directory: &Path) -> bool {
        self.files.iter().all(|file| self.holds(directory, file))
            && self.sources().all(SourceFile::is_unchanged)
            && interrupted(directory).is_ok_and(|path| path.symlink_metadata().is_err())
    }

    fn holds(&self, directory: &Path, file: &ImportedFile) -> bool {
        let path = directory.join(&file.name);
        match &file.origin {
            FileOrigin::Derived { written } => written.holds(&path),
            FileOrigin::Linked { source, named } => {
                self.linked(OsStr::new(&file.name))
                    .is_some_and(|source| source.is_linked_at(&path))
                    && fs::canonicalize(named).is_ok_and(|named| named == *source)
            },
            FileOrigin::InPlace { source } => {
                fs::canonicalize(&path).is_ok_and(|path| path == *source)
            },
        }
    }

    /// The source the import linked under `name`, if it made that link.
    #[must_use]
    pub fn linked(&self, name: &OsStr) -> Option<&SourceFile> {
        let source = self.files.iter().find_map(|file| match &file.origin {
            FileOrigin::Linked { source, .. } if OsStr::new(&file.name) == name => Some(source),
            _ => None,
        })?;
        self.sources().find(|recorded| recorded.path == *source)
    }

    /// The files of `directory` the import made, derived or linked.
    pub(super) fn made(&self) -> impl Iterator<Item = &ImportedFile> {
        self.files
            .iter()
            .filter(|file| !matches!(file.origin, FileOrigin::InPlace { .. }))
    }

    /// Whether `path`, named `name`, is a file the import made and still
    /// is: its link to the source it records, or the derived file it wrote.
    /// Names are compared in any case, as an import claims them: on a
    /// case-insensitive file system both spellings are one file.
    pub(super) fn made_at(&self, name: &OsStr, path: &Path) -> bool {
        let name = name.to_string_lossy().to_lowercase();
        let named = |file: &&ImportedFile| file.name.to_lowercase() == name;
        self.files.iter().filter(named).any(|file| match &file.origin {
            FileOrigin::Derived { written } => written.is_at(path),
            FileOrigin::Linked { source, .. } => self
                .sources()
                .find(|recorded| recorded.path == *source)
                .is_some_and(|source| source.is_linked_at(path)),
            FileOrigin::InPlace { .. } => false,
        })
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

/// Whether `name` is one file directly inside a directory.
fn is_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}
