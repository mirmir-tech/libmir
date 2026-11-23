//! A bounds-checked little-endian reader over a header prefix. Running out of
//! bytes is reported with the length needed, so a caller can read more of the
//! file (or fetch a larger HTTP range) and parse again.

use crate::error::ModelsError;

/// Why a header prefix could not be parsed.
#[derive(Debug)]
pub enum ParseError {
    /// The prefix ends before the header does; at least `needed` bytes are
    /// required.
    Truncated {
        needed: u64,
    },
    Invalid(ModelsError),
}

impl From<ModelsError> for ParseError {
    fn from(error: ModelsError) -> Self {
        Self::Invalid(error)
    }
}

pub type Parsed<T> = Result<T, ParseError>;

pub fn invalid(message: impl Into<String>) -> ParseError {
    ParseError::Invalid(ModelsError::InvalidGguf(message.into()))
}

pub struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub fn take(&mut self, length: u64) -> Parsed<&'a [u8]> {
        let start = self.position;
        let end = usize::try_from(length)
            .ok()
            .and_then(|length| start.checked_add(length))
            .ok_or_else(|| invalid("a length does not fit in memory"))?;
        let slice = self.bytes.get(start..end).ok_or_else(|| ParseError::Truncated {
            needed: u64::try_from(end).unwrap_or(u64::MAX),
        })?;
        self.position = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Parsed<[u8; N]> {
        let bytes = self.take(N as u64)?;
        let mut array = [0_u8; N];
        array.copy_from_slice(bytes);
        Ok(array)
    }

    pub fn u8(&mut self) -> Parsed<u8> {
        Ok(self.array::<1>()?[0])
    }

    pub fn u16(&mut self) -> Parsed<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    pub fn u32(&mut self) -> Parsed<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub fn u64(&mut self) -> Parsed<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub fn i8(&mut self) -> Parsed<i8> {
        Ok(i8::from_le_bytes(self.array()?))
    }

    pub fn i16(&mut self) -> Parsed<i16> {
        Ok(i16::from_le_bytes(self.array()?))
    }

    pub fn i32(&mut self) -> Parsed<i32> {
        Ok(i32::from_le_bytes(self.array()?))
    }

    pub fn i64(&mut self) -> Parsed<i64> {
        Ok(i64::from_le_bytes(self.array()?))
    }

    pub fn f32(&mut self) -> Parsed<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    pub fn f64(&mut self) -> Parsed<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    pub fn bool(&mut self) -> Parsed<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(invalid(format!("boolean byte {other} is neither 0 nor 1"))),
        }
    }

    /// A `u64` length followed by UTF-8 bytes.
    pub fn string(&mut self) -> Parsed<String> {
        let length = self.u64()?;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| invalid("a string is not valid UTF-8"))
    }

    /// A count of items that each take at least `min_bytes`, refused when the
    /// header could never hold that many; a short prefix asks for more bytes.
    pub fn count(&mut self, what: &str, min_bytes: u64, file_len: Option<u64>) -> Parsed<usize> {
        let count = self.u64()?;
        let needed = count
            .checked_mul(min_bytes)
            .and_then(|bytes| bytes.checked_add(u64::try_from(self.position).ok()?))
            .ok_or_else(|| invalid(format!("{what} count {count} overflows")))?;
        if file_len.is_some_and(|length| needed > length) {
            return Err(invalid(format!("{what} count {count} exceeds the file")));
        }
        if needed > u64::try_from(self.bytes.len()).unwrap_or(u64::MAX) {
            return Err(ParseError::Truncated { needed });
        }
        usize::try_from(count).map_err(|_| invalid(format!("{what} count {count} overflows")))
    }
}
