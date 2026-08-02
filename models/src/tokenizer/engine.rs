use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

use super::{TokenizerAssets, bpe, metadata, policy::TokenizerPolicy, tokenized};
use crate::{
    error::{ModelsError, Result},
    generation::GenerationConfig,
    layout::ModelLayout,
};

#[derive(Debug, Clone)]
pub struct TokenizedPrompt {
    pub token_ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingSide {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerInfo {
    pub path: PathBuf,
    pub kind: TokenizerKind,
    pub vocab_size: usize,
    pub stop_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub padding_side: PaddingSide,
    pub default_max_length: Option<usize>,
    pub model_max_length: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    SentencePieceModel,
    TokenizerJson,
    BpeVocab,
}

pub struct TextTokenizer {
    pub(super) inner: Tokenizer,
    path: PathBuf,
    kind: TokenizerKind,
    pub(super) added_tokens: BTreeMap<String, u32>,
    pub(super) configured_stop_token_ids: Vec<u32>,
    pub(super) eos_token_ids: Vec<u32>,
    pub(super) pad_token_id: Option<u32>,
    policy: TokenizerPolicy,
}

impl TextTokenizer {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let assets = TokenizerAssets::from_layout(layout)?;
        let path = layout.root.join(&assets.primary);
        let configured_stop_token_ids =
            GenerationConfig::from_layout(layout)?.stop_token_ids().to_vec();
        let (mut inner, kind) = if assets.kind == TokenizerKind::BpeVocab {
            let merges_path = assets
                .merges
                .as_deref()
                .map(|merges| layout.root.join(merges))
                .ok_or_else(|| ModelsError::MissingFile(layout.root.join("merges.txt")))?;
            let add_prefix_space = bpe::add_prefix_space(layout.tokenizer_config_path.as_deref())?;
            (
                bpe::tokenizer_from_files(&path, &merges_path, add_prefix_space)?,
                TokenizerKind::BpeVocab,
            )
        } else if assets.kind == TokenizerKind::SentencePieceModel {
            (
                super::sentencepiece::tokenizer_from_file(&path)?,
                TokenizerKind::SentencePieceModel,
            )
        } else {
            (Tokenizer::from_file(&path)?, TokenizerKind::TokenizerJson)
        };
        let metadata = metadata::configure(
            &mut inner,
            layout.tokenizer_config_path.as_deref(),
            layout.added_tokens_path.as_deref(),
            layout.special_tokens_map_path.as_deref(),
        )?;
        let policy = TokenizerPolicy::read(layout.tokenizer_config_path.as_deref())?;
        let added_tokens = metadata::inventory(&inner);
        let pad_token_id = policy.pad_token.as_deref().and_then(|token| inner.token_to_id(token));
        Ok(Self {
            inner,
            path,
            kind,
            added_tokens,
            configured_stop_token_ids,
            eos_token_ids: metadata.eos_token_ids,
            pad_token_id,
            policy,
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
            pad_token_id: None,
            policy: TokenizerPolicy::read(None)?,
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
        Ok(tokenized(&encoding, text.len()))
    }

    pub fn encode_with_limit(&self, text: &str, max_length: usize) -> Result<TokenizedPrompt> {
        let tokenizer = self.with_limit(max_length)?;
        Ok(tokenized(&tokenizer.encode(text, true)?, text.len()))
    }

    pub fn encode_pair(
        &self,
        first: &str,
        second: &str,
        max_length: usize,
    ) -> Result<TokenizedPrompt> {
        let tokenizer = self.with_limit(max_length)?;
        Ok(tokenized(
            &tokenizer.encode((first, second), true)?,
            first.len().saturating_add(second.len()),
        ))
    }

    fn with_limit(&self, max_length: usize) -> Result<Tokenizer> {
        let mut tokenizer = self.inner.clone();
        let _configured = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: self.policy.truncation_direction,
        }))?;
        Ok(tokenizer)
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
            pad_token_id: self.pad_token_id,
            padding_side: self.policy.padding_side,
            default_max_length: self.policy.default_max_length,
            model_max_length: self.policy.model_max_length,
        }
    }

    #[must_use]
    pub const fn padding_side(&self) -> PaddingSide {
        self.policy.padding_side
    }

    #[must_use]
    pub const fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    #[must_use]
    pub const fn default_max_length(&self) -> Option<usize> {
        self.policy.default_max_length
    }

    #[must_use]
    pub const fn model_max_length(&self) -> Option<usize> {
        self.policy.model_max_length
    }
}
