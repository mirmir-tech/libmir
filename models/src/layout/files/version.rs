//! Reading a directory as one version of it: read again while it changes
//! under the read, and told as changing where it keeps changing.

use std::{path::Path, time::Duration};

use crate::error::{ModelsError, Result};

/// How many times a directory is read before it is told as changing.
const READS: u32 = 13;

/// How many reads go without a pause; each later one waits twice as long
/// as the one before, from a millisecond: 511 ms in all.
const UNPAUSED: u32 = 4;

/// The pause before read `attempt`, counted from 0.
fn pause(attempt: u32) -> Option<Duration> {
    attempt.checked_sub(UNPAUSED).map(|doubled| Duration::from_millis(1 << doubled))
}

/// What `read` gives of `root` where `version` is the same before and
/// after it, read again, with a `wait` between later reads, until it is;
/// [`ModelsError::ChangedWhileRead`] where it never is. A `version` of
/// `None` tells nothing: the read is then given as it is.
pub(super) fn one_version<V: PartialEq, T>(
    root: &Path,
    version: impl Fn() -> Option<V>,
    mut read: impl FnMut() -> Result<T>,
    mut wait: impl FnMut(Duration),
) -> Result<T> {
    for attempt in 0..READS {
        if let Some(pause) = pause(attempt) {
            wait(pause);
        }
        let before = version();
        let read = read();
        if before.is_none() || version() == before {
            return read;
        }
    }
    Err(ModelsError::ChangedWhileRead(root.to_path_buf()))
}

/// The directory `root` leads to as it stands: its identity and when its
/// entries last changed, which a directory put in its place, or an entry
/// added, removed or renamed in it, changes, and its attributes do not.
#[cfg(unix)]
pub(super) fn of(root: &Path) -> Option<(u64, u64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(root).ok()?;
    Some((metadata.dev(), metadata.ino(), metadata.mtime(), metadata.mtime_nsec()))
}

#[cfg(not(unix))]
pub(super) const fn of(_: &Path) -> Option<()> {
    None
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// Reads `root` as [`one_version`] does, with a directory that changes
    /// during each of the first `changing` reads: what it gave, how many
    /// reads it made, and how long it waited.
    fn read(changing: u32) -> (Result<u32>, u32, Duration) {
        let (reads, waited) = (Cell::new(0), Cell::new(Duration::ZERO));
        let given = one_version(
            Path::new("model"),
            || Some(reads.get().min(changing)),
            || {
                reads.set(reads.get() + 1);
                Ok(reads.get())
            },
            |pause| waited.set(waited.get() + pause),
        );
        (given, reads.get(), waited.get())
    }

    /// A directory that changes under every read is told as changing,
    /// after every read and pause.
    #[test]
    fn a_directory_changing_under_every_read_is_told_as_changing() {
        let (given, reads, waited) = read(u32::MAX);
        assert!(
            matches!(given, Err(ModelsError::ChangedWhileRead(path)) if path == Path::new("model"))
        );
        assert_eq!((reads, waited), (READS, Duration::from_millis(511)));
    }

    /// A directory that settles is given as the first read after, and the
    /// pauses wait only as long as it takes.
    #[test]
    fn a_directory_that_settles_is_read_once_more() -> Result<()> {
        assert_eq!(read(0).0?, 1);
        assert_eq!(read(0).2, Duration::ZERO);
        assert_eq!(read(3).0?, 4);
        let (given, reads, waited) = read(6);
        assert_eq!((given?, reads, waited), (7, 7, Duration::from_millis(1 + 2 + 4)));
        Ok(())
    }

    /// Without a version, as off Unix, the directory is read once.
    #[test]
    fn a_directory_without_a_version_is_read_once() -> Result<()> {
        let reads = Cell::new(0);
        let given = one_version(
            Path::new("model"),
            || None::<()>,
            || {
                reads.set(reads.get() + 1);
                Ok(reads.get())
            },
            |_| {},
        );
        assert_eq!((given?, reads.get()), (1, 1));
        Ok(())
    }
}
