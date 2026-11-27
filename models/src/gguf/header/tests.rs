//! Reading a header from a file: a file shorter than its length said ends
//! the read rather than repeating it, and no read grows beyond the limit.

use super::{
    GgufHeader,
    read::{MAX_HEADER_BYTES, grown},
};
use crate::{
    error::ModelsError,
    gguf::{GgmlType, tests::TestResult},
};

fn header() -> TestResult<Vec<u8>> {
    let mut bytes = crate::gguf::MAGIC.to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    let name = "token_embd.weight";
    bytes.extend(u64::try_from(name.len())?.to_le_bytes());
    bytes.extend(name.as_bytes());
    bytes.extend(1_u32.to_le_bytes());
    bytes.extend(32_u64.to_le_bytes());
    bytes.extend(GgmlType::F32.id().to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.resize(bytes.len().div_ceil(32) * 32 + 128, 0);
    Ok(bytes)
}

#[test]
fn a_whole_file_reads() -> TestResult {
    let bytes = header()?;
    let header = GgufHeader::read_from(bytes.as_slice(), u64::try_from(bytes.len())?)?;
    assert_eq!(header.tensors.len(), 1);
    Ok(())
}

/// A file that shrank after its length was taken ends the read with an
/// error, as does one whose header is cut short.
#[test]
fn a_file_ending_early_is_refused_once() -> TestResult {
    let bytes = header()?;
    let shrunk = GgufHeader::read_from(&bytes[..20], u64::try_from(bytes.len())?);
    assert!(
        matches!(&shrunk, Err(ModelsError::InvalidGguf(message)) if message.contains("ended")),
        "{shrunk:?}"
    );
    let length = 40;
    let truncated = GgufHeader::read_from(&bytes[..length], u64::try_from(length)?);
    assert!(matches!(truncated, Err(ModelsError::InvalidGguf(_))), "{truncated:?}");
    Ok(())
}

#[test]
fn a_read_grows_within_the_limit() {
    let limit = MAX_HEADER_BYTES;
    assert_eq!(grown(limit - 1, limit, limit), Some(limit));
    assert_eq!(grown(16, 20, limit), Some(32));
    assert_eq!(grown(16, 20, 24), Some(24));
    assert_eq!(grown(16, 16, limit), None);
    assert_eq!(grown(16, limit + 1, limit), None);
}

/// Metadata arrays of more elements than the bound are refused before
/// they are materialized, each a `MetadataValue` many times its byte.
#[test]
fn metadata_arrays_are_bounded_by_elements() -> TestResult {
    let array = |count: usize| -> TestResult<Vec<u8>> {
        let mut bytes = crate::gguf::MAGIC.to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        let key = "bytes";
        bytes.extend(u64::try_from(key.len())?.to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(9_u32.to_le_bytes());
        bytes.extend(0_u32.to_le_bytes());
        bytes.extend(u64::try_from(count)?.to_le_bytes());
        bytes.resize(bytes.len() + count, 0);
        Ok(bytes)
    };
    let within = array(4)?;
    assert!(GgufHeader::parse(&within, Some(u64::try_from(within.len())?)).is_ok());
    let beyond = array(crate::gguf::value::MAX_ARRAY_ELEMENTS + 1)?;
    let parsed = GgufHeader::parse(&beyond, Some(u64::try_from(beyond.len())?));
    assert!(matches!(parsed, Err(crate::gguf::ParseError::Invalid(_))));
    Ok(())
}

/// An array declaring more elements than the bound is refused from its
/// count alone, not after the reader grows to fetch its bytes.
#[test]
fn an_oversized_array_is_refused_before_its_bytes_are_read() {
    let mut prefix = crate::gguf::MAGIC.to_vec();
    prefix.extend(3_u32.to_le_bytes());
    prefix.extend(0_u64.to_le_bytes());
    prefix.extend(1_u64.to_le_bytes());
    prefix.extend(5_u64.to_le_bytes());
    prefix.extend(b"bytes");
    prefix.extend(9_u32.to_le_bytes());
    prefix.extend(0_u32.to_le_bytes());
    prefix.extend(900_000_000_u64.to_le_bytes());
    let parsed = GgufHeader::parse(&prefix, Some(1 << 31));
    assert!(matches!(parsed, Err(crate::gguf::ParseError::Invalid(_))));
}
