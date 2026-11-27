use std::path::PathBuf;

use thiserror::Error;
use tokenizers::Error as TokenizerError;

pub type Result<T> = std::result::Result<T, ModelsError>;

#[derive(Debug, Error)]
pub enum ModelsError {
    #[error("this model does not declare a supported reasoning template control")]
    UnsupportedReasoningControl,
    #[error("invalid GGUF: {0}")]
    InvalidGguf(String),
    #[error("GGML tensor type {0} is not supported")]
    UnsupportedGgmlType(u32),
    #[error("model file is missing: {0}")]
    MissingFile(PathBuf),
    #[error("cannot import a GGUF model to {}: {conflict}", path.display())]
    ImportConflict {
        path: PathBuf,
        conflict: crate::gguf::ImportConflict,
    },
    #[error("the GGUF import in {} was cut short while it was written; write it again", .0.display())]
    InterruptedImport(PathBuf),
    /// The directory was replaced or changed every time it was read, as a
    /// GGUF import republished meanwhile: what was read may mix versions.
    #[error("the model in {} changed while it was read, as when it is republished; load it again", .0.display())]
    ChangedWhileRead(PathBuf),
    #[error("the import manifest {} names {name:?}, not a file of its directory", manifest.display())]
    InvalidImportEntry { manifest: PathBuf, name: String },
    #[error("invalid model config: {0}")]
    InvalidConfig(String),
    #[error("invalid safetensors payload range: {0}")]
    InvalidTensorRange(String),
    #[error("invalid safetensors header: {0}")]
    InvalidSafeTensorsHeader(String),
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
