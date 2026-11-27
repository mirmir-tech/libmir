//! Publishing an import in one step: it is written to `.<name>.staging`
//! beside its directory and, once complete, exchanged with the directory
//! in one rename (`renameat2` with `RENAME_EXCHANGE` on Linux,
//! `renameatx_np` with `RENAME_SWAP` on Apple systems), so every lookup of
//! a path in the directory finds a whole import, the earlier or the new
//! one; the earlier one, then at `.staging`, is removed. Where the exchange
//! is refused, by a file system, kernel or sandbox without it, the
//! directory moves to `.<name>.previous` and the import into its place, a
//! moment in which the directory is missing, and `.previous` is removed
//! after. Publishers of one directory take turns through a lock on
//! `.<name>.lock`, and one finding the directory gone while `.previous`
//! remains restores it first. A directory that is a symlink is replaced as
//! the link: the import becomes a plain directory, the link is removed, and
//! the directory it linked to is never changed.
//!
//! Each lookup sees one version, not each reader: one that reads several
//! files by path may find some in the earlier import and some in the new.
//! [`ModelLayout::inspect`](crate::layout::ModelLayout::inspect) reads
//! again when the directory changed meanwhile, but a load that inspects
//! and later opens the files it found, or inspects twice, may find a later
//! import than the one it inspected.

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
    let mut replaced = directory.symlink_metadata().is_ok();
    if replaced {
        match exchange(&staging, directory) {
            // `staging` holds the earlier import, or the link that stood
            // for it, which is removed as a link.
            Exchange::Done => return remove_stale(&staging),
            Exchange::Vanished => replaced = false,
            Exchange::Refused => {
                if let Err(error) = fs::rename(directory, &previous) {
                    drop(remove_stale(&staging));
                    return Err(error.into());
                }
            },
        }
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

/// How an exchange of a complete import with its directory went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Exchange {
    /// The two changed places.
    Done,
    /// One of them was gone, as the directory removed meanwhile; neither
    /// moved.
    Vanished,
    /// The system, the file system or a sandbox refused; neither moved, so
    /// two renames replace the directory instead.
    Refused,
}

#[cfg(test)]
thread_local! {
    /// Whether this thread's exchanges are refused, as where the system
    /// cannot exchange two entries.
    pub(super) static NO_EXCHANGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// How this thread's exchanges went, in order.
    pub(super) static EXCHANGES: std::cell::RefCell<Vec<Exchange>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Exchanges the entries `left` and `right` name, links not followed, in
/// one step. Any refusal leaves both in place, whatever its cause: a file
/// system or kernel without the exchange, or a sandbox's filter.
pub(super) fn exchange(left: &Path, right: &Path) -> Exchange {
    let exchanged = if refused() {
        Err(io::ErrorKind::Unsupported.into())
    } else {
        swap(left, right)
    };
    let exchange = match exchanged {
        Ok(()) => Exchange::Done,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Exchange::Vanished,
        Err(_) => Exchange::Refused,
    };
    #[cfg(test)]
    EXCHANGES.with_borrow_mut(|exchanges| exchanges.push(exchange));
    exchange
}

#[cfg(test)]
fn refused() -> bool {
    NO_EXCHANGE.get()
}

#[cfg(not(test))]
const fn refused() -> bool {
    false
}

/// `renameat2` with `RENAME_EXCHANGE` on Linux, `renameatx_np` with
/// `RENAME_SWAP` on Apple systems.
#[cfg(any(target_os = "android", target_os = "linux", target_vendor = "apple"))]
fn swap(left: &Path, right: &Path) -> io::Result<()> {
    use rustix::fs::{CWD, RenameFlags, renameat_with};
    Ok(renameat_with(CWD, left, CWD, right, RenameFlags::EXCHANGE)?)
}

/// Elsewhere the system exchanges no entries.
#[cfg(not(any(target_os = "android", target_os = "linux", target_vendor = "apple")))]
fn swap(_: &Path, _: &Path) -> io::Result<()> {
    Err(io::ErrorKind::Unsupported.into())
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
#[expect(clippy::unnecessary_wraps, reason = "fails on Unix, where any existing path resolves")]
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
