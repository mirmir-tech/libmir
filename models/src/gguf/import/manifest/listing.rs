//! Which files of a directory a loader reads: in a GGUF import, of the
//! names an import writes, only those its manifest holds, so a user's file
//! the import kept beside it, such as a processor configuration of a
//! model it no longer holds, is not read as the import's. Its weights are
//! the shards and assembled files the manifest holds, never those a
//! `SafeTensors` index left there names. A directory is the import's only
//! while its `config.json` is what the import wrote: a checkpoint written
//! over it is read as that checkpoint. An import whose write was cut short
//! is not read at all.

use std::{collections::BTreeSet, path::Path};

use super::{FileOrigin, ImportManifest, interrupted};
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

/// The derived file that tells the model a directory holds.
const CONFIG: &str = DERIVED[0];

/// The index of a `SafeTensors` checkpoint, which names its weights: an
/// import never writes one, and its weights are those its manifest holds.
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

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
        let imported = ImportManifest::describing(directory)
            .map(|manifest| manifest.files.into_iter().map(|file| file.name).collect());
        Ok(Self { imported })
    }

    /// Whether a loader reads the file `name`: any where no import is, else
    /// one the import holds or of a name that does not tell the model: no
    /// derived file, weight file or `SafeTensors` index.
    #[must_use]
    pub fn reads(&self, name: &str) -> bool {
        let written = DERIVED.contains(&name)
            || name == SAFETENSORS_INDEX
            || matches!(
                WeightContainer::of(Path::new(name)),
                Some(WeightContainer::SafeTensors | WeightContainer::Gguf)
            );
        self.imported.as_ref().is_none_or(|names| !written || names.contains(name))
    }
}

impl ImportManifest {
    /// The manifest of the import whose model `directory` holds, which
    /// decides what a loader reads there: `None` without one, and where its
    /// `config.json` is not the one it wrote, as when a checkpoint was
    /// downloaded over the import, or an earlier version's manifest cannot
    /// tell. The directory then loads as it is, as a stale import of
    /// unchanged files still loads as the import.
    #[must_use]
    pub fn describing(directory: &Path) -> Option<Self> {
        let config = directory.join(CONFIG);
        Self::read(directory).filter(|manifest| {
            manifest.files.iter().any(|file| {
                file.name == CONFIG
                    && matches!(
                        &file.origin,
                        FileOrigin::Derived { written: Some(written) } if written.holds(&config)
                    )
            })
        })
    }
}
