use std::{fs, path::Path};

use serde_json::Value;
use tokenizers::TruncationDirection;

use super::PaddingSide;
use crate::error::Result;

pub(super) struct TokenizerPolicy {
    pub padding_side: PaddingSide,
    pub truncation_direction: TruncationDirection,
    pub pad_token: Option<String>,
    pub default_max_length: Option<usize>,
    pub model_max_length: Option<usize>,
}

impl TokenizerPolicy {
    pub(super) fn read(path: Option<&Path>) -> Result<Self> {
        let value = path
            .map(fs::read_to_string)
            .transpose()?
            .map(|json| serde_json::from_str::<Value>(&json))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            padding_side: match value.get("padding_side").and_then(Value::as_str) {
                Some("left") => PaddingSide::Left,
                _ => PaddingSide::Right,
            },
            truncation_direction: match value.get("truncation_side").and_then(Value::as_str) {
                Some("left") => TruncationDirection::Left,
                _ => TruncationDirection::Right,
            },
            pad_token: string_token(&value, "pad_token"),
            default_max_length: usize_value(&value, "max_length")?,
            model_max_length: usize_value(&value, "model_max_length")?,
        })
    }
}

fn string_token(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(|value| match value {
        Value::String(value) => Some(value.clone()),
        Value::Object(value) => value.get("content").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    })
}

fn usize_value(value: &Value, field: &str) -> Result<Option<usize>> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|value| Ok(usize::try_from(value)?))
        .transpose()
}
