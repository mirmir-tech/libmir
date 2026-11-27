//! Whether two paths name one file, by the identity the file system gives
//! it: its device and inode on Unix, its volume and file index on Windows
//! (through `same-file`, which reads them with `GetFileInformationByHandle`).
//! A path whose identity cannot be read is no file's.

use std::path::Path;

/// Whether `left` and `right` resolve to one file, links followed: a hard
/// link and the file it links, or two spellings of one name.
pub(super) fn same_file(left: &Path, right: &Path) -> bool {
    imp::same_file(left, right)
}

/// Whether `left` and `right` are one entry, links not followed: a symlink
/// is never the file it links to. On Unix an entry is its inode, which hard
/// links share; elsewhere it is its name, so two hard links are two.
pub(super) fn same_entry(left: &Path, right: &Path) -> bool {
    imp::same_entry(left, right)
}

#[cfg(unix)]
mod imp {
    use std::{fs::Metadata, os::unix::fs::MetadataExt, path::Path};

    fn identity(metadata: &Metadata) -> (u64, u64) {
        (metadata.dev(), metadata.ino())
    }

    pub(super) fn same_file(left: &Path, right: &Path) -> bool {
        match (left.metadata(), right.metadata()) {
            (Ok(left), Ok(right)) => identity(&left) == identity(&right),
            _ => false,
        }
    }

    pub(super) fn same_entry(left: &Path, right: &Path) -> bool {
        match (left.symlink_metadata(), right.symlink_metadata()) {
            (Ok(left), Ok(right)) => identity(&left) == identity(&right),
            _ => false,
        }
    }
}

#[cfg(not(unix))]
mod imp {
    use std::{fs, path::Path};

    pub(super) fn same_file(left: &Path, right: &Path) -> bool {
        same_file::is_same_file(left, right).unwrap_or(false)
    }

    /// Opening a path follows a link, and two hard links share a file's
    /// identity, so an entry has none of its own to read: two paths are one
    /// entry where they spell one name, in any case as such systems compare
    /// names, in one directory, both there and of one kind, and a link then
    /// to one target.
    pub(super) fn same_entry(left: &Path, right: &Path) -> bool {
        let (Ok(left_kind), Ok(right_kind)) = (left.symlink_metadata(), right.symlink_metadata())
        else {
            return false;
        };
        let name = |path: &Path| path.file_name().map(|name| name.to_string_lossy().to_lowercase());
        let parent = |path: &Path| path.parent().map(Path::to_path_buf);
        let links = match (left_kind.is_symlink(), right_kind.is_symlink()) {
            (false, false) => true,
            (true, true) => fs::read_link(left)
                .is_ok_and(|target| fs::read_link(right).is_ok_and(|other| other == target)),
            _ => false,
        };
        links
            && name(left).is_some()
            && name(left) == name(right)
            && parent(left)
                .zip(parent(right))
                .is_some_and(|(left, right)| same_file(&left, &right))
    }
}
