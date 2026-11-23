//! Tiktoken schemes: the split pattern and special-token block a Python
//! tokenizer defines in code rather than in a data file.

use std::{collections::HashMap, fs, path::Path};

use serde_json::Value;

use crate::error::{ModelsError, Result};

/// A tiktoken split and special-token scheme libmir rebuilds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TikTokenScheme {
    /// [`HAN_RUNS_CASED_WORDS`] with 256 special tokens after the ranks.
    HanRunsCasedWords,
}

/// Han runs split off first, then the o200k word classes with Han
/// excluded: cased words with contractions, up to three digits,
/// punctuation, newlines and spaces. Source: `pat_str` of
/// `tokenization_moonshot.py`.
pub const HAN_RUNS_CASED_WORDS: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

impl TikTokenScheme {
    /// The scheme the checkpoint's `tokenizer_config.json` class defines.
    pub fn from_config(path: Option<&Path>) -> Result<Self> {
        let config = config(path)?;
        match config.get("tokenizer_class").and_then(Value::as_str) {
            Some("TikTokenTokenizer") => Ok(Self::HanRunsCasedWords),
            other => Err(ModelsError::InvalidConfig(format!(
                "no supported tiktoken scheme for tokenizer class {other:?}"
            ))),
        }
    }

    pub const fn pattern(self) -> &'static str {
        match self {
            Self::HanRunsCasedWords => HAN_RUNS_CASED_WORDS,
        }
    }

    /// Special tokens that follow the ranks. The defining Python tokenizer
    /// reserves two more past them, which the embeddings do not cover, so
    /// no model can take them.
    pub const fn special_count(self) -> u32 {
        match self {
            Self::HanRunsCasedWords => 256,
        }
    }
}

/// Names of the special tokens the configuration declares, by id.
pub fn declared_special_tokens(path: Option<&Path>) -> Result<HashMap<u32, String>> {
    let config = config(path)?;
    let Some(decoder) = config.get("added_tokens_decoder").and_then(Value::as_object) else {
        return Ok(HashMap::new());
    };
    decoder
        .iter()
        .map(|(id, token)| {
            let content = token
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| ModelsError::InvalidConfig("added token has no content".into()))?;
            Ok((id.parse::<u32>()?, content.to_owned()))
        })
        .collect()
}

fn config(path: Option<&Path>) -> Result<Value> {
    let path = path.ok_or_else(|| {
        ModelsError::InvalidConfig("a tiktoken checkpoint needs its tokenizer_config.json".into())
    })?;
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}
