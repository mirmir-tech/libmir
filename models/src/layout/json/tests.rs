use serde_json::json;

use super::{parse, parse_bytes};
use crate::error::Result;

/// NVIDIA's Nemotron-H configuration, as Transformers writes it.
#[test]
fn reads_pythons_non_finite_literals_as_encoded_floats() -> Result<()> {
    let value = parse(r#"{"time_step_limit": [0.0, Infinity], "low": -Infinity, "x": NaN}"#)?;
    assert_eq!(value["time_step_limit"], json!([0.0, {"__float__": "Infinity"}]));
    assert_eq!(value["low"], json!({"__float__": "-Infinity"}));
    assert_eq!(value["x"], json!({"__float__": "NaN"}));
    Ok(())
}

/// A literal inside a string is text, and a document broken otherwise
/// keeps its own error.
#[test]
fn leaves_strings_and_other_errors_alone() -> Result<()> {
    let value = parse(r#"{"name": "Infinity \" NaN", "limit": Infinity}"#)?;
    assert_eq!(value["name"], json!("Infinity \" NaN"));
    assert!(parse(r#"{"a": }"#).is_err());
    assert!(parse(r#"{"a": Infinity,}"#).is_err());
    assert_eq!(parse_bytes(b"[Infinity]")?, json!([{"__float__": "Infinity"}]));
    assert!(parse_bytes(b"[\xff]").is_err());
    Ok(())
}
