//! Manifests earlier versions wrote, without a format: one recording each
//! link's source and the path it was given, before it one naming each
//! file's origin alone, and first one naming the files alone. Those
//! versions could record a user's file as their own link, and wrote their
//! derived files over a user's, so a file is read as the import's only
//! where the record shows the import made it and it still is that file;
//! anything else, the user's or not known to be the import's, is kept as
//! found in place. A file found in place whose source the record does not
//! tell leaves the import stale, so the next write records it.

use std::{ffi::OsStr, fs, path::Path};

use serde::Deserialize;

use super::{FileOrigin, ImportManifest, ImportedFile, SourceFile};
use crate::layout::WeightContainer;

#[derive(Deserialize)]
struct Manifest<File> {
    text: Vec<SourceFile>,
    vision: Vec<SourceFile>,
    files: Vec<File>,
}

#[derive(Deserialize)]
struct Entry<Origin> {
    name: String,
    origin: Origin,
}

/// An origin as the version recording links' sources wrote it.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Sourced {
    Derived,
    Linked {
        source: std::path::PathBuf,
        named: std::path::PathBuf,
    },
    InPlace,
}

/// An origin as the version before it wrote it, without a link's source.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Named {
    Derived,
    Linked,
    InPlace,
}

/// An earlier manifest in `bytes`, with each file's origin as `directory`
/// shows it now; `None` for none of the earlier forms.
pub(super) fn read(bytes: &[u8], directory: &Path) -> Option<ImportManifest> {
    if let Ok(manifest) = serde_json::from_slice::<Manifest<Entry<Sourced>>>(bytes) {
        return Some(convert(manifest, |sources, name, origin| match origin {
            Sourced::Derived => derived(),
            Sourced::Linked { source, named } if made_by_import(directory, name, &named) => {
                FileOrigin::Linked { source, named }
            },
            // Whoever made it, its given path is not known to still resolve
            // to its source, which only a new write records.
            Sourced::Linked { .. } => FileOrigin::InPlace { source: None },
            Sourced::InPlace => found_in_place(sources, directory, name),
        }));
    }
    if let Ok(manifest) = serde_json::from_slice::<Manifest<Entry<Named>>>(bytes) {
        return Some(convert(manifest, |sources, name, origin| match origin {
            Named::Derived => derived(),
            // A link without the path it was given cannot be told from a
            // user's link given as the shard, which that version replaced.
            Named::Linked | Named::InPlace => found_in_place(sources, directory, name),
        }));
    }
    let Manifest { text, vision, files } =
        serde_json::from_slice::<Manifest<String>>(bytes).ok()?;
    let files = files.into_iter().map(|name| Entry { name, origin: () }).collect();
    Some(convert(Manifest { text, vision, files }, |sources, name, ()| {
        if is_derived(name, sources) {
            derived()
        } else {
            found_in_place(sources, directory, name)
        }
    }))
}

/// A file an earlier version recorded as derived, by name alone: the
/// import's only while it holds what a new import writes there, which
/// replacing it does not change; so a user's file moved there since, with
/// whatever time, is kept.
const fn derived() -> FileOrigin {
    FileOrigin::Derived { written: None }
}

/// `manifest` with each file's `origin` as `current` reads it, given the
/// recorded sources.
fn convert<Origin>(
    manifest: Manifest<Entry<Origin>>,
    current: impl Fn(&[&SourceFile], &str, Origin) -> FileOrigin,
) -> ImportManifest {
    let Manifest { text, vision, files } = manifest;
    let sources = text.iter().chain(&vision).collect::<Vec<_>>();
    let files = files
        .into_iter()
        .map(|Entry { name, origin }| {
            let origin = current(&sources, &name, origin);
            ImportedFile { name, origin }
        })
        .collect();
    ImportManifest { text, vision, files }
}

/// Whether an earlier version's record of `name` as its link to the shard
/// it was given at `named` shows the import made it: `named`, under the
/// same name, is still a file of another directory than `directory`, and
/// that directory is no import. Where it was the entry itself, a user's
/// file that version replaced with its link, it is not, nor where that
/// cannot be told any more: `named` or its directory gone, as after the
/// import was moved, or `named` an entry of an import, as of the one this
/// is a copy of.
fn made_by_import(directory: &Path, name: &str, named: &Path) -> bool {
    let (Ok(directory), Some(parent)) = (fs::canonicalize(directory), named.parent()) else {
        return false;
    };
    named.file_name() == Some(OsStr::new(name))
        && named.symlink_metadata().is_ok()
        && parent.join(super::MANIFEST).symlink_metadata().is_err()
        && fs::canonicalize(parent).is_ok_and(|parent| parent != directory)
}

/// `name` as a file found in place, tied to the source it is: a recorded
/// source at its own path. A link resolving to one could have resolved to
/// another when the import was made, so its source is not told.
fn found_in_place(sources: &[&SourceFile], directory: &Path, name: &str) -> FileOrigin {
    let path = directory.join(name);
    let itself = super::is_file_name(name)
        && path.symlink_metadata().is_ok_and(|metadata| !metadata.file_type().is_symlink());
    let source = itself
        .then(|| fs::canonicalize(&path).ok())
        .flatten()
        .filter(|path| sources.iter().any(|source| source.path == *path));
    FileOrigin::InPlace { source }
}

/// Whether `name` is one the import derives, and no source's.
fn is_derived(name: &str, sources: &[&SourceFile]) -> bool {
    !sources.iter().any(|source| source.path.file_name() == Some(name.as_ref()))
        && (super::listing::DERIVED.contains(&name)
            || WeightContainer::of(Path::new(name)) == Some(WeightContainer::SafeTensors))
}
