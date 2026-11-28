use std::path::Path;

use tokenizers::Tokenizer;

use super::{TokenizerAssets, TokenizerKind, bpe};
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

pub(super) fn open_disk_tokenizer(
    layout: &ModelLayout,
    assets: &TokenizerAssets,
    path: &Path,
) -> Result<(Tokenizer, TokenizerKind)> {
    match assets.kind {
        TokenizerKind::BpeVocab => {
            let merges_path = assets
                .merges
                .as_deref()
                .map(|merges| layout.root.join(merges))
                .ok_or_else(|| ModelsError::MissingFile(layout.root.join("merges.txt")))?;
            let add_prefix_space = bpe::add_prefix_space(layout.tokenizer_config_path.as_deref())?;
            Ok((
                bpe::tokenizer_from_files(path, &merges_path, add_prefix_space)?,
                TokenizerKind::BpeVocab,
            ))
        },
        TokenizerKind::TikToken => {
            let config = layout.tokenizer_config_path.as_deref();
            Ok((super::tiktoken::tokenizer_from_files(path, config)?, TokenizerKind::TikToken))
        },
        TokenizerKind::SentencePieceModel => Ok((
            super::sentencepiece::tokenizer_from_file(path)?,
            TokenizerKind::SentencePieceModel,
        )),
        TokenizerKind::TokenizerJson => {
            Ok((Tokenizer::from_file(path)?, TokenizerKind::TokenizerJson))
        },
    }
}
