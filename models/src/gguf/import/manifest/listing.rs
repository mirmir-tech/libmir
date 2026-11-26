//! Which files of a directory a loader reads: in a GGUF import, of the
//! names an import writes, only those its manifest holds, so a user's file
//! the import kept beside it, such as a processor configuration of a
//! model it no longer holds, is not read as the import's. An import whose
//! write was cut short is not read at all.

use std::{collections::BTreeSet, path::Path};

use super::{ImportManifest, interrupted};
use crate::{
    error::{ModelsError, Result},
    layout::WeightContainer,
};

/// The files an import derives from GGUF metadata, beside the `SafeTensors`
/// a vision import assembles.
pub(super) const DERIVED: [&str; 5] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
];

/// The files of a directory a loader reads, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing {
    /// The files of the import the directory holds; `None` for no import.
    imported: Option<BTreeSet<String>>,
}

impl Listing {
    /// The listing of `directory`, refused while a write of an import there
    /// is cut short: its files may be half the earlier model's.
    pub fn of(directory: &Path) -> Result<Self> {
        if interrupted(directory)?.symlink_metadata().is_ok() {
            return Err(ModelsError::InterruptedImport(directory.to_path_buf()));
        }
        let imported = ImportManifest::read(directory)
            .map(|manifest| manifest.files.into_iter().map(|file| file.name).collect());
        Ok(Self { imported })
    }

    /// Whether a loader reads the file `name`: any where no import is, else
    /// one the import holds or of a name no import writes.
    #[must_use]
    pub fn reads(&self, name: &str) -> bool {
        let written = DERIVED.contains(&name)
            || matches!(
                WeightContainer::of(Path::new(name)),
                Some(WeightContainer::SafeTensors | WeightContainer::Gguf)
            );
        self.imported.as_ref().is_none_or(|names| !written || names.contains(name))
    }
}
