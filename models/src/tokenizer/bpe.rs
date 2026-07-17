use std::{fs, path::Path};

use tokenizers::{Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

use crate::error::Result;

pub(super) fn tokenizer_from_files(
    vocab_path: &Path,
    merges_path: &Path,
    add_prefix_space: bool,
) -> Result<Tokenizer> {
    let model = BPE::builder()
        .files(
            vocab_path.to_string_lossy().into_owned(),
            merges_path.to_string_lossy().into_owned(),
        )
        .build()?;
    let byte_level = ByteLevel::new(add_prefix_space, true, true);
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));
    Ok(tokenizer)
}

pub(super) fn add_prefix_space(path: Option<&Path>) -> Result<bool> {
    let Some(path) = path else {
        return Ok(false);
    };
    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(value
        .get("add_prefix_space")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false))
}
