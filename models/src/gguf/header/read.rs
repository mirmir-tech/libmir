//! Reading a local file's header: a first prefix, grown until the header
//! fits, from one open file. Each read must reach as far as the file's
//! length said, so a file that shrinks meanwhile is refused rather than read
//! again, and no read goes beyond [`MAX_HEADER_BYTES`].

use std::{fs::File, io::Read, path::Path};

use super::GgufHeader;
use crate::{
    error::{ModelsError, Result},
    gguf::reader::ParseError,
};

/// The first read of a local file; a larger header is read on from there.
const FIRST_READ: u64 = 16 * 1024 * 1024;
/// Vocabulary arrays make headers large, but not this large.
pub(super) const MAX_HEADER_BYTES: u64 = 1024 * 1024 * 1024;

impl GgufHeader {
    /// Reads and parses the header of a local file, growing the read until the
    /// header fits.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        #[cfg(test)]
        reads::count(path);
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Self::read_from(file, file_len).map_err(|error| match error {
            ModelsError::InvalidGguf(error) => {
                ModelsError::InvalidGguf(format!("{}: {error}", path.display()))
            },
            other => other,
        })
    }

    /// Parses the header of a file of `file_len` bytes read from `file`.
    pub(in crate::gguf) fn read_from(mut file: impl Read, file_len: u64) -> Result<Self> {
        let limit = file_len.min(MAX_HEADER_BYTES);
        let mut bytes = Vec::new();
        let mut wanted = FIRST_READ.min(limit);
        loop {
            let read = u64::try_from(bytes.len())?;
            bytes.reserve_exact(usize::try_from(wanted - read)?);
            let _ = (&mut file).take(wanted - read).read_to_end(&mut bytes)?;
            if u64::try_from(bytes.len())? < wanted {
                return Err(ModelsError::InvalidGguf(format!(
                    "the file ended after {} of its {file_len} bytes while its header was read",
                    bytes.len()
                )));
            }
            match Self::parse(&bytes, Some(file_len)) {
                Ok(header) => return Ok(header),
                Err(ParseError::Invalid(error)) => return Err(error),
                Err(ParseError::Truncated { needed }) => {
                    wanted = grown(wanted, needed, limit).ok_or_else(|| {
                        ModelsError::InvalidGguf(if needed > MAX_HEADER_BYTES {
                            format!("the header needs {needed} bytes, beyond {MAX_HEADER_BYTES}")
                        } else {
                            "the header is truncated".to_owned()
                        })
                    })?;
                },
            }
        }
    }
}

/// The next read after `wanted` bytes left a header `needed` bytes long:
/// at least doubled, within `limit`; `None` where `needed` is beyond it or
/// no longer than what was read.
pub(super) fn grown(wanted: u64, needed: u64, limit: u64) -> Option<u64> {
    (needed <= limit && needed > wanted).then(|| needed.max(wanted.saturating_mul(2)).min(limit))
}

/// How many times this thread read each file's header, for tests that
/// bound the reads of a loading phase.
#[cfg(test)]
pub(in crate::gguf) mod reads {
    use std::{
        cell::RefCell,
        collections::BTreeMap,
        path::{Path, PathBuf},
    };

    thread_local! {
        static READS: RefCell<BTreeMap<PathBuf, usize>> = const { RefCell::new(BTreeMap::new()) };
    }

    pub(in crate::gguf) fn count(path: &Path) {
        READS.with(|reads| *reads.borrow_mut().entry(path.to_owned()).or_default() += 1);
    }

    /// The reads since the last call, by file.
    pub(in crate::gguf) fn take() -> BTreeMap<PathBuf, usize> {
        READS.with(|reads| std::mem::take(&mut *reads.borrow_mut()))
    }
}
