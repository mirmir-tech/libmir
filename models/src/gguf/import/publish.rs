//! Publishing an import in one step: it is written to `.<name>.staging`
//! beside its directory, which moves to `.<name>.previous` only once the new
//! import is complete and is removed after the swap. Publishers of one
//! directory take turns through a lock on `.<name>.lock`, and one finding
//! the directory gone while `.previous` remains restores it first. A
//! directory that is a symlink is replaced as the link: the import becomes
//! a plain directory, and the directory it linked to is never changed.

use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

use super::{
    ImportConflict,
    files::{beside, remove_stale},
};
use crate::error::{ModelsError, Result};

/// Replaces `directory` with what `fill` writes into a fresh directory. A
/// failure of `fill` leaves `directory` as it was; `sources`, the files the
/// import links, must not live inside `directory`, by their path or the
/// file it resolves to.
pub(super) fn replace(
    directory: &Path,
    sources: &[PathBuf],
    fill: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let directory = &placed(directory)?;
    let _turn = lock(directory)?;
    let (staging, previous) = (beside(directory, "staging")?, beside(directory, "previous")?);
    // Interrupted between the two renames: the last good import is aside.
    if directory.symlink_metadata().is_err() && previous.symlink_metadata().is_ok() {
        fs::rename(&previous, directory)?;
    }
    // Publishing replaces or removes all three directories, and whatever
    // lives in them: a user's link there given as a shard included, though
    // it resolves elsewhere.
    let replaced = [directory, &staging, &previous]
        .into_iter()
        .flat_map(|path| [std::path::absolute(path).ok(), fs::canonicalize(path).ok()])
        .flatten()
        .collect::<Vec<_>>();
    for source in sources {
        let given = std::path::absolute(source)?;
        let placed = given
            .parent()
            .and_then(|parent| fs::canonicalize(parent).ok())
            .zip(given.file_name())
            .map(|(parent, name)| parent.join(name));
        let paths = [Some(given), placed, Some(fs::canonicalize(source)?)];
        if paths
            .iter()
            .flatten()
            .any(|path| replaced.iter().any(|inside| path.starts_with(inside)))
        {
            return Err(ModelsError::ImportConflict {
                path: directory.clone(),
                conflict: ImportConflict::SourceInside,
            });
        }
    }
    remove_stale(&staging)?;
    remove_stale(&previous)?;
    if let Err(error) = fill(&staging) {
        drop(remove_stale(&staging));
        return Err(error);
    }
    let replaced = directory.symlink_metadata().is_ok();
    if replaced {
        fs::rename(directory, &previous)?;
    }
    if let Err(error) = fs::rename(&staging, directory) {
        if replaced {
            drop(fs::rename(&previous, directory));
        }
        drop(remove_stale(&staging));
        return Err(error.into());
    }
    remove_stale(&previous)
}

/// `directory` as publishing replaces it: absolute, every existing
/// directory above it resolved, links followed, and its own entry, a link
/// included, named as given. Every spelling of one entry is one path,
/// beside which its lock, staging and backup are. A destination that names
/// no entry of its own, as `.`, `..`, `x/..` or a root, is refused: it
/// would replace whatever directory it resolves to, through any link.
pub(super) fn placed(directory: &Path) -> Result<PathBuf> {
    let unnamed = || ModelsError::ImportConflict {
        path: directory.to_path_buf(),
        conflict: ImportConflict::UnnamedDestination,
    };
    let Some(Component::Normal(name)) = directory.components().next_back() else {
        return Err(unnamed());
    };
    let absolute = std::path::absolute(directory)?;
    let parent = absolute.parent().ok_or_else(unnamed)?;
    Ok(resolved(parent)?.join(name))
}

/// `directory` as writing fills it: absolute, its existing part resolved,
/// links followed, the directories still to be made named as given. A
/// write through a link writes, and takes turns with others, where it
/// leads, and goes on there should the link change meanwhile.
pub(super) fn resolved(directory: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(directory)?;
    let mut missing = Vec::new();
    let mut existing = absolute.as_path();
    loop {
        let found = match fs::canonicalize(existing) {
            Ok(path) => path,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // A missing part is made as named; `..` past one is not
                // resolved.
                let (Some(parent), Some(name)) = (existing.parent(), existing.file_name()) else {
                    return Err(error.into());
                };
                missing.push(name);
                existing = parent;
                continue;
            },
            Err(error) => unresolved(existing, error)?,
        };
        let mut path = found;
        path.extend(missing.iter().rev());
        return Ok(path);
    }
}

/// An existing path that cannot be resolved: an error on Unix.
#[cfg(unix)]
fn unresolved(_: &Path, error: io::Error) -> Result<PathBuf> {
    Err(error.into())
}

/// An existing path that cannot be resolved, as on a volume that tells no
/// final path (some network and virtual drives on Windows), as it is
/// spelled, absolute.
#[cfg(not(unix))]
fn unresolved(existing: &Path, _: io::Error) -> Result<PathBuf> {
    Ok(existing.to_path_buf())
}

/// Waits for, and holds until dropped, the lock of publishing or writing
/// `directory`, given [`placed`] or [`resolved`]; the system releases it if
/// the process dies.
pub(super) fn lock(directory: &Path) -> Result<fs::File> {
    let path = beside(directory, "lock")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::options().create(true).truncate(false).write(true).open(path)?;
    file.lock()?;
    Ok(file)
}
