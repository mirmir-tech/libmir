use std::{collections::BTreeMap, path::PathBuf};

use super::TokenizerKind;
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

const TOKENIZER_JSON: &str = "tokenizer.json";
const TOKENIZER_MODEL: &str = "tokenizer.model";
const VOCAB: &str = "vocab.json";
const MERGES: &str = "merges.txt";
const OPTIONAL: [&str; 3] =
    ["tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerAssets {
    pub kind: TokenizerKind,
    pub primary: String,
    pub merges: Option<String>,
    pub metadata: Vec<String>,
    pub total_bytes: u64,
}

impl TokenizerAssets {
    pub fn discover(files: &BTreeMap<String, u64>) -> Result<Self> {
        let (kind, primary, merges) = if files.contains_key(TOKENIZER_JSON) {
            (TokenizerKind::TokenizerJson, TOKENIZER_JSON, None)
        } else if files.contains_key(TOKENIZER_MODEL) {
            (TokenizerKind::SentencePieceModel, TOKENIZER_MODEL, None)
        } else if files.contains_key(VOCAB) {
            if !files.contains_key(MERGES) {
                return Err(ModelsError::MissingFile(PathBuf::from(MERGES)));
            }
            (TokenizerKind::BpeVocab, VOCAB, Some(MERGES.to_owned()))
        } else {
            return Err(ModelsError::MissingFile(PathBuf::from(TOKENIZER_JSON)));
        };
        let metadata = OPTIONAL
            .into_iter()
            .filter(|name| files.contains_key(*name))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let total_bytes = std::iter::once(primary)
            .chain(merges.as_deref())
            .chain(metadata.iter().map(String::as_str))
            .filter_map(|name| files.get(name))
            .try_fold(0_u64, |total, bytes| total.checked_add(*bytes))
            .ok_or_else(|| ModelsError::InvalidConfig("tokenizer asset size overflow".into()))?;
        Ok(Self {
            kind,
            primary: primary.into(),
            merges,
            metadata,
            total_bytes,
        })
    }

    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let paths = [
            layout.tokenizer_path.as_ref(),
            layout.vocab_path.as_ref(),
            layout.merges_path.as_ref(),
            layout.tokenizer_config_path.as_ref(),
            layout.added_tokens_path.as_ref(),
            layout.special_tokens_map_path.as_ref(),
        ];
        let mut files = BTreeMap::new();
        for path in paths.into_iter().flatten() {
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                let _previous = files.insert(name.to_owned(), std::fs::metadata(path)?.len());
            }
        }
        Self::discover(&files)
    }
}

#[cfg(test)]
mod tests;
