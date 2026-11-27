//! Which files of a directory an import made and still are what it made:
//! those its manifest records, the manifest of a later write cut short, or
//! the records that write carried, with their link to the recorded source
//! or the contents of the derived file. Only those are replaced or removed;
//! every other file is the user's: kept, found in place where it is the
//! shard, else refused, naming it.

use std::{collections::BTreeSet, ffi::OsStr, fs, io, path::Path};

use super::super::{ImportConflict, ImportManifest};
use crate::error::{ModelsError, Result};

/// The manifests telling what earlier writes of a directory made.
pub(super) struct Records {
    /// The last complete import's, then a later write's cut short and the
    /// records it carried.
    manifests: Vec<ImportManifest>,
}

impl Records {
    pub(super) fn load(directory: &Path) -> Result<Self> {
        let mut manifests = ImportManifest::load(directory)?.into_iter().collect::<Vec<_>>();
        manifests.extend(ImportManifest::load_interrupted(directory)?);
        Ok(Self { manifests })
    }

    /// Whether `path`, named `name`, is a file an import made and still is,
    /// a derived one whose contents were not recorded if it holds
    /// `writing`.
    fn made(&self, name: &OsStr, path: &Path, writing: Option<&[u8]>) -> bool {
        self.manifests.iter().any(|manifest| manifest.made_at(name, path, writing))
    }

    /// The records of what earlier writes made that is still in
    /// `directory`, for a new write to carry until it has removed or
    /// replaced it.
    pub(super) fn standing(&self, directory: &Path) -> Vec<&ImportManifest> {
        self.manifests
            .iter()
            .filter(|manifest| {
                manifest.made().any(|file| {
                    let name = OsStr::new(&file.name);
                    manifest.made_at(name, &directory.join(name), None)
                })
            })
            .collect()
    }

    /// Refuses to write `contents` as the derived file `name` over a file
    /// at `path` that no import wrote as it is now.
    pub(super) fn writable(&self, name: &OsStr, path: &Path, contents: &[u8]) -> Result<()> {
        if !exists(path)? || self.made(name, path, Some(contents)) {
            return Ok(());
        }
        Err(ModelsError::ImportConflict {
            path: path.to_path_buf(),
            conflict: ImportConflict::OccupiedFile,
        })
    }

    /// Whether `path` already holds the shard `target`, found in place. An
    /// import's own file there, its link or a derived file, is replaced;
    /// any other file is the user's: kept, found in place where it resolves
    /// to `target`, the source itself or a user's link to it, else refused.
    pub(super) fn present(&self, name: &OsStr, path: &Path, target: &Path) -> Result<bool> {
        if !exists(path)? || self.made(name, path, None) {
            return Ok(false);
        }
        if fs::canonicalize(path).is_ok_and(|resolved| resolved == target) {
            return Ok(true);
        }
        Err(ModelsError::ImportConflict {
            path: path.to_path_buf(),
            conflict: ImportConflict::OccupiedLink,
        })
    }

    /// The names of what the earlier writes made, but the import holding
    /// `held` does not hold under that very name.
    pub(super) fn made_but(&self, held: &BTreeSet<String>) -> Vec<String> {
        let mut stale = Vec::new();
        for file in self.manifests.iter().flat_map(ImportManifest::made) {
            if !held.contains(&file.name) && !stale.contains(&file.name) {
                stale.push(file.name.clone());
            }
        }
        stale
    }

    /// Removes `path`, named `name`, while it is still a file an import
    /// made; else it is the user's and stays.
    pub(super) fn remove_made(&self, name: &OsStr, path: &Path) -> Result<()> {
        if self.made(name, path, None) {
            match fs::remove_file(path) {
                Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error.into()),
                _ => {},
            }
        }
        Ok(())
    }
}

/// Whether anything, a dangling link included, is at `path`.
fn exists(path: &Path) -> Result<bool> {
    match path.symlink_metadata() {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}
