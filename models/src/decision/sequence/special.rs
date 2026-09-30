use std::{fs, path::Path};

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::error::{ModelsError, Result};

/// Special tokens named by `tokenizer_config.json`, resolved to ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SpecialTokens {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    /// Text of the mask token; it is blanked out of every input so user text
    /// cannot forge an option marker.
    pub mask_text: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(rename = "cls_token")]
    cls: TokenName,
    #[serde(rename = "sep_token")]
    sep: TokenName,
    #[serde(rename = "mask_token")]
    mask: TokenName,
}

/// A special token written either as its text or as an added-token object.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenName {
    Text(String),
    Added { content: String },
}

impl TokenName {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) | Self::Added { content: text } => text,
        }
    }
}

impl SpecialTokens {
    pub fn resolve(tokenizer: &Tokenizer, config: &Path) -> Result<Self> {
        let config: TokenizerConfig = serde_json::from_str(&fs::read_to_string(config)?)?;
        let id = |text: &str| {
            tokenizer.token_to_id(text).ok_or_else(|| {
                ModelsError::InvalidConfig(format!(
                    "special token `{text}` is not in the vocabulary"
                ))
            })
        };
        let mask_text = config.mask.into_text();
        Ok(Self {
            cls: id(&config.cls.into_text())?,
            sep: id(&config.sep.into_text())?,
            mask: id(&mask_text)?,
            mask_text,
        })
    }
}
