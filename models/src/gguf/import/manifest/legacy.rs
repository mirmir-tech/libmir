//! The manifest an earlier version wrote, which named the import's files
//! without how each came there.

use std::{ffi::OsStr, path::Path};

use serde::Deserialize;

use super::{FileOrigin, ImportManifest, ImportedFile, SourceFile, is_file_name};
use crate::layout::WeightContainer;

/// The files an import derives from GGUF metadata, beside the `SafeTensors`
/// a vision import assembles.
const DERIVED: [&str; 5] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
];

#[derive(Deserialize)]
pub(super) struct Manifest {
    text: Vec<SourceFile>,
    vision: Vec<SourceFile>,
    files: Vec<String>,
}

impl Manifest {
    /// The manifest with the origin each file shows in `directory` now. A
    /// source's name is the import's link only while it still links to that
    /// source; any other file there, the source itself included, is taken as
    /// found in place. A name the import derives is its own, and any other
    /// is taken as not the import's, so nothing the import did not make is
    /// ever replaced or removed.
    pub(super) fn typed(self, directory: &Path) -> ImportManifest {
        let Self { text, vision, files } = self;
        let files = files
            .into_iter()
            .map(|name| {
                let source = text
                    .iter()
                    .chain(&vision)
                    .find(|source| source.path.file_name() == Some(OsStr::new(&name)));
                let origin = match source {
                    Some(source)
                        if is_file_name(&name) && source.is_linked_at(&directory.join(&name)) =>
                    {
                        FileOrigin::Linked
                    },
                    None if is_derived(&name) => FileOrigin::Derived,
                    _ => FileOrigin::InPlace,
                };
                ImportedFile { name, origin }
            })
            .collect();
        ImportManifest { text, vision, files }
    }
}

fn is_derived(name: &str) -> bool {
    DERIVED.contains(&name)
        || WeightContainer::of(Path::new(name)) == Some(WeightContainer::SafeTensors)
}
