//! Checkpoint configuration as Transformers writes it.
//!
//! Python's `json` writes non-finite floats as the bare literals `Infinity`,
//! `-Infinity` and `NaN`, which JSON lacks (NVIDIA's Nemotron-H
//! `time_step_limit` is `[0.0, Infinity]`). They are read as the `{"__float__":
//! "Infinity"}` encoding the parsers already take for non-finite values.

use std::{fs, path::Path};

use serde_json::Value;

use crate::error::Result;

/// The JSON at `path`, its non-finite literals encoded.
///
/// # Errors
///
/// When `path` cannot be read or is not JSON.
pub fn read(path: impl AsRef<Path>) -> Result<Value> {
    Ok(parse(&fs::read_to_string(path)?)?)
}

/// `source` parsed, its non-finite literals encoded.
///
/// # Errors
///
/// When `source` is not JSON with Python's non-finite literals.
pub fn parse(source: &str) -> serde_json::Result<Value> {
    serde_json::from_str(source).or_else(|error| {
        encode_non_finite(source).map_or(Err(error), |encoded| serde_json::from_str(&encoded))
    })
}

/// [`parse`] of UTF-8 `bytes`, as fetched from the Hub.
///
/// # Errors
///
/// When `bytes` are not UTF-8 JSON with Python's non-finite literals.
pub fn parse_bytes(bytes: &[u8]) -> serde_json::Result<Value> {
    // Not UTF-8, serde_json says where.
    std::str::from_utf8(bytes).map_or_else(|_| serde_json::from_slice(bytes), parse)
}

/// Python's non-finite literals, as `__float__` objects.
const LITERALS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];

/// `source` with each non-finite literal outside a string encoded; none
/// when it has none, so a document that is broken otherwise keeps its
/// own error.
fn encode_non_finite(source: &str) -> Option<String> {
    let mut encoded = String::with_capacity(source.len());
    let (mut in_string, mut escaped, mut found) = (false, false, false);
    let mut rest = source;
    while let Some(next) = rest.chars().next() {
        if in_string {
            match next {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {},
            }
        } else if next == '"' {
            in_string = true;
        } else if let Some(literal) = LITERALS.iter().find(|literal| rest.starts_with(**literal)) {
            encoded.push_str(r#"{"__float__":""#);
            encoded.push_str(literal);
            encoded.push_str(r#""}"#);
            rest = &rest[literal.len()..];
            found = true;
            continue;
        }
        encoded.push(next);
        rest = &rest[next.len_utf8()..];
    }
    found.then_some(encoded)
}

#[cfg(test)]
mod tests;
