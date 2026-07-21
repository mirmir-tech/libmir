use std::path::PathBuf;

use thiserror::Error;
use tokenizers::Error as TokenizerError;

pub type Result<T> = std::result::Result<T, ModelsError>;

#[derive(Debug, Error)]
pub enum ModelsError {
    #[error("model file is missing: {0}")]
    MissingFile(PathBuf),
    #[error("invalid model config: {0}")]
    InvalidConfig(String),
    #[error("invalid safetensors payload range: {0}")]
    InvalidTensorRange(String),
    #[error("invalid model integer: {0}")]
    InvalidInteger(#[from] std::num::TryFromIntError),
    #[error("invalid tokenizer token id: {0}")]
    TokenId(#[from] std::num::ParseIntError),
    #[error("invalid model float: {0}")]
    InvalidFloat(#[from] std::num::ParseFloatError),
    #[error("invalid model text: {0}")]
    InvalidText(#[from] std::str::Utf8Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("TOML model specification error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("cannot serialize TOML model specification: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("chat template error: {0}")]
    Template(#[from] minijinja::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("BPE tokenizer error: {0}")]
    Bpe(#[from] tokenizers::models::bpe::Error),
    #[error("unsupported tokenizer format: {path} ({reason})")]
    UnsupportedTokenizer { path: PathBuf, reason: String },
}
