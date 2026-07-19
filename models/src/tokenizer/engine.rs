use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use tokenizers::Tokenizer;

use super::{bpe, metadata};
use crate::{
    error::{ModelsError, Result},
    generation::GenerationConfig,
    layout::ModelLayout,
};

#[derive(Debug, Clone)]
pub struct TokenizedPrompt {
    pub token_ids: Vec<u32>,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerInfo {
    pub path: PathBuf,
    pub kind: TokenizerKind,
    pub vocab_size: usize,
    pub stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    SentencePieceModel,
    TokenizerJson,
    BpeVocab,
}

pub struct TextTokenizer {
    inner: Tokenizer,
    path: PathBuf,
    kind: TokenizerKind,
    added_tokens: BTreeMap<String, u32>,
    configured_stop_token_ids: Vec<u32>,
    eos_token_ids: Vec<u32>,
}

impl TextTokenizer {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let path = layout
            .tokenizer_path
            .as_deref()
            .ok_or_else(|| ModelsError::MissingFile(layout.root.join("tokenizer.json")))?;
        let configured_stop_token_ids =
            GenerationConfig::from_layout(layout)?.stop_token_ids().to_vec();
        let (mut inner, kind) = if layout.vocab_path.as_deref() == Some(path) {
            let merges_path = layout
                .merges_path
                .as_deref()
                .ok_or_else(|| ModelsError::MissingFile(layout.root.join("merges.txt")))?;
            let add_prefix_space = bpe::add_prefix_space(layout.tokenizer_config_path.as_deref())?;
            (
                bpe::tokenizer_from_files(path, merges_path, add_prefix_space)?,
                TokenizerKind::BpeVocab,
            )
        } else if path.extension().is_some_and(|ext| ext == "model") {
            (
                super::sentencepiece::tokenizer_from_file(path)?,
                TokenizerKind::SentencePieceModel,
            )
        } else {
            (Tokenizer::from_file(path)?, TokenizerKind::TokenizerJson)
        };
        let metadata = metadata::configure(
            &mut inner,
            layout.tokenizer_config_path.as_deref(),
            layout.added_tokens_path.as_deref(),
            layout.special_tokens_map_path.as_deref(),
        )?;
        let added_tokens = metadata::inventory(&inner);
        Ok(Self {
            inner,
            path: path.to_path_buf(),
            kind,
            added_tokens,
            configured_stop_token_ids,
            eos_token_ids: metadata.eos_token_ids,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = Tokenizer::from_file(&path)?;
        Ok(Self {
            added_tokens: metadata::inventory(&inner),
            inner,
            path,
            kind: TokenizerKind::TokenizerJson,
            configured_stop_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<TokenizedPrompt> {
        self.encode_with_special_tokens(text, true)
    }

    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedPrompt> {
        let encoding = self.inner.encode(text, add_special_tokens)?;
        Ok(TokenizedPrompt {
            token_ids: encoding.get_ids().to_vec(),
            bytes: text.len(),
        })
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        Ok(self.inner.decode(token_ids, true)?)
    }

    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    #[must_use]
    pub fn token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    #[must_use]
    pub fn added_token_id(&self, token: &str) -> Option<u32> {
        self.added_tokens.get(token).copied()
    }

    #[must_use]
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = self.configured_stop_token_ids.clone();
        for id in &self.eos_token_ids {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        for token in [
            "<eos>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<end_of_turn>",
            "<turn|>",
            "<|eot_id|>",
            "<|tool_response>",
        ] {
            if let Some(id) = self.inner.token_to_id(token)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        ids
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    #[must_use]
    pub fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            path: self.path.clone(),
            kind: self.kind,
            vocab_size: self.vocab_size(),
            stop_token_ids: self.stop_token_ids(),
        }
    }
}
