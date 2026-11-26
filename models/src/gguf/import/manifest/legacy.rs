//! Manifests earlier versions wrote: one naming each file's origin without
//! the source of a link, and before it one naming the files alone. Each is
//! read with what the directory shows now, so nothing the import did not
//! make is ever replaced or removed.

use std::{ffi::OsStr, fs, path::Path};

use serde::Deserialize;

use super::{FileOrigin, ImportManifest, ImportedFile, SourceFile};
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
struct Manifest<File> {
    text: Vec<SourceFile>,
    vision: Vec<SourceFile>,
    files: Vec<File>,
}

/// A file of a manifest that named origins without a link's source.
#[derive(Deserialize)]
struct Named {
    name: String,
    origin: Origin,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Origin {
    Derived,
    Linked,
    InPlace,
}

/// An earlier manifest in `bytes`, with each file's origin as `directory`
/// shows it now; `None` for neither earlier form.
pub(super) fn read(bytes: &[u8], directory: &Path) -> Option<ImportManifest> {
    if let Ok(Manifest { text, vision, files }) = serde_json::from_slice::<Manifest<Named>>(bytes) {
        let files = files
            .into_iter()
            .map(|Named { name, origin }| {
                let origin = match origin {
                    Origin::Derived => FileOrigin::Derived,
                    Origin::Linked => linked(&text, &vision, directory, &name),
                    Origin::InPlace => FileOrigin::InPlace,
                };
                ImportedFile { name, origin }
            })
            .collect();
        return Some(ImportManifest { text, vision, files });
    }
    let Manifest { text, vision, files } =
        serde_json::from_slice::<Manifest<String>>(bytes).ok()?;
    let files = files
        .into_iter()
        .map(|name| {
            let origin = match linked(&text, &vision, directory, &name) {
                FileOrigin::InPlace if is_derived(&name, &text, &vision) => FileOrigin::Derived,
                origin => origin,
            };
            ImportedFile { name, origin }
        })
        .collect();
    Some(ImportManifest { text, vision, files })
}

/// Takes back for the user each entry an earlier version recorded as its
/// link though the path it was given is that very entry: the user's file
/// in `directory`, a symlink included, which that version replaced with
/// its own link to the source. Found in place, it is never the import's to
/// replace or remove.
pub(super) fn own_entries(manifest: &mut ImportManifest, directory: &Path) {
    let Ok(directory) = fs::canonicalize(directory) else {
        return;
    };
    for file in &mut manifest.files {
        if let FileOrigin::Linked { named, .. } = &file.origin
            && named.file_name() == Some(OsStr::new(&file.name))
            && named.parent().and_then(|parent| fs::canonicalize(parent).ok())
                == Some(directory.clone())
        {
            file.origin = FileOrigin::InPlace;
        }
    }
}

/// `name`'s origin by what it is now: a link to the source it still links,
/// whatever either is named, else a file not the import's, the source
/// itself found in place included.
fn linked(text: &[SourceFile], vision: &[SourceFile], directory: &Path, name: &str) -> FileOrigin {
    let path = directory.join(name);
    if !super::is_file_name(name) {
        return FileOrigin::InPlace;
    }
    text.iter().chain(vision).find(|source| source.is_linked_at(&path)).map_or(
        FileOrigin::InPlace,
        |source| FileOrigin::Linked {
            // The path it was given is unknown: the source itself.
            source: source.path.clone(),
            named: source.path.clone(),
        },
    )
}

/// Whether `name` is one the import derives, and no source's.
fn is_derived(name: &str, text: &[SourceFile], vision: &[SourceFile]) -> bool {
    let named = |source: &&SourceFile| source.path.file_name() == Some(name.as_ref());
    !text.iter().chain(vision).any(|source| named(&source))
        && (DERIVED.contains(&name)
            || WeightContainer::of(Path::new(name)) == Some(WeightContainer::SafeTensors))
}
