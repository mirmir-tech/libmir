//! The files of an import, resolved and checked before the first write, and
//! each written whole: to a partial file renamed over its name.

use std::{
    collections::BTreeSet,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
};

use super::{
    FileOrigin, Import, ImportConflict, ImportManifest, ImportedFile, MANIFEST, SourceFile,
};
use crate::error::{ModelsError, Result};

/// What writing an import into one directory does.
pub(super) struct Plan {
    /// `(name, contents)`, `config.json` last.
    derived: Vec<(String, Vec<u8>)>,
    links: Vec<Link>,
    /// What the previous import made that this one does not hold.
    stale: Vec<Stale>,
    /// Written after everything else.
    manifest: ImportManifest,
}

/// A file a previous import made, by name, and the source of its link.
struct Stale {
    name: String,
    link: Option<SourceFile>,
}

/// A shard linked under its own name.
struct Link {
    name: OsString,
    target: PathBuf,
    /// The shard itself is already there, under its name.
    present: bool,
}

impl Plan {
    pub(super) fn new(import: &Import, directory: &Path) -> Result<Self> {
        let derived = import.derived()?;
        let mut names = BTreeSet::new();
        for (name, _) in &derived {
            claim(&mut names, OsStr::new(name), directory)?;
        }
        claim(&mut names, OsStr::new(MANIFEST), directory)?;
        // The previous import's manifest tells the links it made, which may
        // be hard links where symlinks are unavailable, from sources it
        // found in place.
        let previous = ImportManifest::read(directory);
        let mut links = Vec::new();
        let mut sources = [Vec::new(), Vec::new()];
        for (shards, recorded) in
            [&import.shards, &import.vision_shards].into_iter().zip(&mut sources)
        {
            for shard in shards {
                let name = file_name(shard)?;
                claim(&mut names, name, directory)?;
                let target = fs::canonicalize(shard)?;
                let owned = previous.as_ref().and_then(|manifest| manifest.linked(name));
                let present = present(&directory.join(name), &target, owned)?;
                recorded.push(SourceFile::of(&target)?);
                links.push(Link { name: name.to_owned(), target, present });
            }
        }
        let stale = previous
            .iter()
            .flat_map(ImportManifest::made)
            .filter(|file| !names.contains(&file.name.to_lowercase()))
            .map(|file| Stale {
                name: file.name.clone(),
                link: previous
                    .as_ref()
                    .and_then(|manifest| manifest.linked(OsStr::new(&file.name)))
                    .cloned(),
            })
            .collect();
        let files = derived
            .iter()
            .map(|(name, _)| {
                Ok(ImportedFile {
                    name: name.clone(),
                    origin: FileOrigin::Derived,
                })
            })
            .chain(links.iter().map(|link| {
                let name = link.name.to_str().map(str::to_owned).ok_or_else(|| {
                    ModelsError::InvalidGguf(format!("{} is not UTF-8", link.name.display()))
                })?;
                let origin = if link.present {
                    FileOrigin::InPlace
                } else {
                    FileOrigin::Linked
                };
                Ok(ImportedFile { name, origin })
            }))
            .collect::<Result<Vec<_>>>()?;
        let [text, vision] = sources;
        Ok(Self {
            derived,
            links,
            stale,
            manifest: ImportManifest { text, vision, files },
        })
    }

    pub(super) fn write(&self, directory: &Path) -> Result<()> {
        // A model switched for another leaves nothing of the old one that a
        // loader would find; a file no import made is never touched.
        for stale in &self.stale {
            remove_made(&directory.join(&stale.name), stale.link.as_ref())?;
        }
        for link in self.links.iter().filter(|link| !link.present) {
            let path = directory.join(&link.name);
            let partial = beside(&path, "partial")?;
            remove_stale(&partial)?;
            link_file(&link.target, &partial)?;
            fs::rename(&partial, &path)?;
        }
        for (name, bytes) in &self.derived {
            let path = directory.join(name);
            let partial = beside(&path, "partial")?;
            fs::write(&partial, bytes)?;
            fs::rename(&partial, &path)?;
        }
        let path = directory.join(MANIFEST);
        let partial = beside(&path, "partial")?;
        fs::write(&partial, serde_json::to_vec_pretty(&self.manifest)?)?;
        fs::rename(&partial, &path)?;
        Ok(())
    }
}

/// Takes `name` for one file of the import; names differing only in case
/// would be one file on a case-insensitive file system.
fn claim(names: &mut BTreeSet<String>, name: &OsStr, directory: &Path) -> Result<()> {
    if names.insert(name.to_string_lossy().to_lowercase()) {
        return Ok(());
    }
    Err(ModelsError::ImportConflict {
        path: directory.join(name),
        conflict: ImportConflict::DuplicateName,
    })
}

/// Whether `path` is already the shard `target` itself. A link there, a
/// symlink or the hard link a previous import made to its `owned` source,
/// is replaced; any other file, one found in place included, is kept and
/// refused.
fn present(path: &Path, target: &Path, owned: Option<&SourceFile>) -> Result<bool> {
    let metadata = match path.symlink_metadata() {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        return Ok(false);
    }
    if fs::canonicalize(path)? == target {
        return Ok(true);
    }
    if owned.is_some_and(|source| source.is_linked_at(path)) {
        return Ok(false);
    }
    Err(ModelsError::ImportConflict {
        path: path.to_path_buf(),
        conflict: ImportConflict::OccupiedLink,
    })
}

/// Removes `path`, a file an import made: derived, or its link to `link`'s
/// source, when it is still that link.
fn remove_made(path: &Path, link: Option<&SourceFile>) -> Result<()> {
    let metadata = match path.symlink_metadata() {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let made = link.map_or_else(
        || metadata.is_file(),
        |source| metadata.file_type().is_symlink() || source.is_linked_at(path),
    );
    if made {
        fs::remove_file(path)?;
    }
    Ok(())
}

/// The final component of `path`.
pub(super) fn file_name(path: &Path) -> Result<&OsStr> {
    path.file_name()
        .ok_or_else(|| ModelsError::InvalidGguf(format!("{} has no file name", path.display())))
}

/// `.<name>.<role>` beside `path`: its partial write, or a published
/// directory's staging, backup and lock.
pub(super) fn beside(path: &Path, role: &str) -> Result<PathBuf> {
    let mut sibling = OsString::from(".");
    sibling.push(file_name(path)?);
    sibling.push(".");
    sibling.push(role);
    Ok(path.with_file_name(sibling))
}

/// Removes a file or directory left by an interrupted write, if any.
pub(super) fn remove_stale(path: &Path) -> Result<()> {
    let result = match path.symlink_metadata() {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => Err(error),
    };
    Ok(result?)
}

#[cfg(unix)]
fn link_file(target: &Path, link: &Path) -> Result<()> {
    Ok(std::os::unix::fs::symlink(target, link)?)
}

#[cfg(not(unix))]
fn link_file(target: &Path, link: &Path) -> Result<()> {
    Ok(fs::hard_link(target, link)?)
}
