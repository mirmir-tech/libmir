//! A source file of an import as the import saw it, a file the import
//! wrote as it wrote it, and how a file in the import is told to be the
//! source, the import's link to it, or still what the import wrote.

use std::{
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A source file as the import saw it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    /// Its canonical path.
    pub path: PathBuf,
    pub bytes: u64,
    /// Nanoseconds since the Unix epoch, when the file system reports them.
    pub modified: Option<u64>,
}

/// A file the import wrote, by its contents.
///
/// A file with these contents is still the import's to replace or remove,
/// whatever its time, as a copy, a move or a file system reporting times
/// coarsely keeps it; any other file there is the user's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Written {
    pub bytes: u64,
    /// The BLAKE3 digest of its contents, in hexadecimal.
    pub digest: String,
    /// The file's identity once in place, which, unchanged, tells it holds
    /// these contents without reading them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stamp: Option<Stamp>,
}

/// A file's identity by its metadata.
///
/// Its times of modification and of change and, where the system tells
/// them, its device and inode: any write, replacement or copy changes one
/// of them, and only then is the file read to tell what it holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
    pub modified: Option<u64>,
    pub changed: u64,
    pub device: u64,
    pub inode: u64,
}

impl Stamp {
    /// The identity of the regular file at `path`, where the system tells
    /// all of it.
    #[cfg(unix)]
    #[must_use]
    pub fn of(path: &Path) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        let metadata = path.symlink_metadata().ok().filter(fs::Metadata::is_file)?;
        let seconds = u64::try_from(metadata.ctime()).ok()?;
        let nanos = u64::try_from(metadata.ctime_nsec()).ok()?;
        Some(Self {
            modified: modified(&metadata),
            changed: seconds.checked_mul(1_000_000_000)?.checked_add(nanos)?,
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    /// Without a change time and inode, a file is always read.
    #[cfg(not(unix))]
    #[must_use]
    pub const fn of(_: &Path) -> Option<Self> {
        None
    }
}

impl Written {
    /// The file of `contents`, before it is in place.
    pub fn of(contents: &[u8]) -> Result<Self> {
        Ok(Self {
            bytes: u64::try_from(contents.len())?,
            digest: digest(contents),
            stamp: None,
        })
    }

    /// Whether `path` is a regular file of the contents the import wrote,
    /// read only unless its identity is as recorded.
    #[must_use]
    pub fn holds(&self, path: &Path) -> bool {
        (self.stamp.is_some() && Stamp::of(path) == self.stamp) || self.is_at(path)
    }

    /// Whether `path` is a regular file of the contents the import wrote,
    /// always read: to replace or remove it.
    #[must_use]
    pub fn is_at(&self, path: &Path) -> bool {
        path.symlink_metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == self.bytes)
            && fs::read(path).is_ok_and(|contents| digest(&contents) == self.digest)
    }
}

fn digest(contents: &[u8]) -> String {
    blake3::hash(contents).to_hex().to_string()
}

/// `metadata`'s modification time, in nanoseconds since the Unix epoch.
fn modified(metadata: &fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .and_then(|elapsed| u64::try_from(elapsed.as_nanos()).ok())
}

impl SourceFile {
    /// `path` as it is now.
    pub fn of(path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            bytes: metadata.len(),
            modified: modified(&metadata),
        })
    }

    /// Whether `path` is the link an import made to this source: a symlink
    /// to its path, or a hard link to the very same file elsewhere. A copy,
    /// even one of the same size and time, is not, nor is the source itself
    /// at its own path.
    pub(in crate::gguf::import) fn is_linked_at(&self, path: &Path) -> bool {
        match path.symlink_metadata() {
            Ok(link) if link.file_type().is_symlink() => {
                fs::read_link(path).is_ok_and(|target| target == self.path)
            },
            Ok(_) => !self.is_at(path) && super::super::identity::same_file(&self.path, path),
            Err(_) => false,
        }
    }

    /// Whether `path` is this source at its own path, found in place.
    fn is_at(&self, path: &Path) -> bool {
        fs::canonicalize(path).is_ok_and(|path| path == self.path)
    }

    /// Whether the file is still there with the same size and time.
    #[must_use]
    pub fn is_unchanged(&self) -> bool {
        Self::of(&self.path).is_ok_and(|now| now == *self)
    }
}
