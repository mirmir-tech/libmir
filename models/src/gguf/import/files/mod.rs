//! The files of an import, resolved and checked before the first write, and
//! each written whole: to a partial file renamed over its name. The new
//! manifest is written beside its place before any file it names changes,
//! and moved there last, so a write cut short leaves an import that reads
//! stale, that no loader reads, and whose files are still known to be the
//! import's.

mod ownership;
mod write;
use std::{
    collections::BTreeSet,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
};

use ownership::Records;
#[cfg(test)]
pub(super) use write::{CUT_SHORT, Step};

use super::{Import, ImportConflict, MANIFEST, SourceFile};
use crate::error::{ModelsError, Result};

/// What writing an import into one directory does.
pub(super) struct Plan {
    /// `(name, contents)`, `config.json` last.
    derived: Vec<(String, Vec<u8>)>,
    links: Vec<Link>,
    /// What an earlier import made that this one does not hold, removed
    /// while it still is what that import made.
    stale: Vec<String>,
    records: Records,
    text: Vec<SourceFile>,
    vision: Vec<SourceFile>,
}

/// A shard linked under its own name.
struct Link {
    name: String,
    /// The shard's path as the import was given it, absolute.
    named: PathBuf,
    target: PathBuf,
    /// The shard itself is already there, under its name.
    present: bool,
}

impl Plan {
    pub(super) fn new(import: &Import, directory: &Path) -> Result<Self> {
        let derived = import.derived()?;
        let mut names = BTreeSet::new();
        claim(&mut names, OsStr::new(MANIFEST), directory)?;
        // The earlier imports' manifests tell the files they made, which
        // may be hard links where symlinks are unavailable. Every name they
        // hold is checked before any is used.
        let records = Records::load(directory)?;
        for (name, contents) in &derived {
            claim(&mut names, OsStr::new(name), directory)?;
            records.writable(OsStr::new(name), &directory.join(name), contents)?;
        }
        let mut links = Vec::new();
        let mut sources = [Vec::new(), Vec::new()];
        for (shards, recorded) in
            [&import.shards, &import.vision_shards].into_iter().zip(&mut sources)
        {
            for shard in shards {
                let name = file_name(shard)?;
                claim(&mut names, name, directory)?;
                let target = fs::canonicalize(shard)?;
                let present = is_entry_in(shard, directory)
                    || records.present(name, &directory.join(name), &target)?;
                recorded.push(SourceFile::of(&target)?);
                links.push(Link {
                    name: utf8(name)?,
                    named: std::path::absolute(shard)?,
                    target,
                    present,
                });
            }
        }
        let held = derived
            .iter()
            .map(|(name, _)| name.clone())
            .chain(links.iter().map(|link| link.name.clone()))
            .collect::<BTreeSet<_>>();
        let stale = records.made_but(&held);
        let [text, vision] = sources;
        Ok(Self {
            derived,
            links,
            stale,
            records,
            text,
            vision,
        })
    }
}

/// The partial write beside `path`, clear of any an earlier write left.
fn partial(path: &Path) -> Result<PathBuf> {
    let partial = beside(path, "partial")?;
    remove_partial(&partial)?;
    Ok(partial)
}

/// Removes the file or link an interrupted write left at `path`, a name
/// only writes use; a directory there is not one, and refuses the write.
fn remove_partial(path: &Path) -> Result<()> {
    match path.symlink_metadata() {
        Ok(metadata) if metadata.is_dir() => Err(ModelsError::ImportConflict {
            path: path.to_path_buf(),
            conflict: ImportConflict::OccupiedFile,
        }),
        Ok(_) => Ok(fs::remove_file(path)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Writes `contents` to a new file at `path`, never through a link there.
pub(in crate::gguf::import) fn write_new(path: &Path, contents: &[u8]) -> Result<()> {
    use std::io::Write as _;
    remove_partial(path)?;
    let mut file = fs::File::options().write(true).create_new(true).open(path)?;
    file.write_all(contents)?;
    Ok(())
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

fn utf8(name: &OsStr) -> Result<String> {
    name.to_str()
        .map(str::to_owned)
        .ok_or_else(|| ModelsError::InvalidGguf(format!("{} is not UTF-8", name.display())))
}

/// Whether `shard` is itself the entry under its name in `directory`: a
/// file the user put there, a symlink included, that the import found in
/// place and never replaces or removes, whatever a previous manifest says.
fn is_entry_in(shard: &Path, directory: &Path) -> bool {
    let parent = std::path::absolute(shard)
        .ok()
        .and_then(|shard| fs::canonicalize(shard.parent()?).ok());
    parent.is_some() && parent == fs::canonicalize(directory).ok()
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
