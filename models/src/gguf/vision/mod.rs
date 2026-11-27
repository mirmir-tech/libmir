//! Vision towers of llama.cpp `mmproj` GGUF files (`general.architecture =
//! clip`), imported as the vision half of a Hugging Face checkpoint: tensor
//! names, `vision_config`, the image processor and, where llama.cpp splits
//! one checkpoint tensor in two, a small `SafeTensors` file holding it whole.

mod gemma;
mod qwen;
mod safetensors;

use std::path::Path;

use serde_json::{Map, Value};

use super::{GgmlType, GgufHeader, GgufModel, MetadataValue};
use crate::error::{ModelsError, Result};

/// The `clip.projector_type` families with an import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projector {
    /// Qwen3-VL and Qwen3.5 spatial merger (`qwen3vl_merger`).
    Qwen3VlMerger,
    /// Gemma 4's pooled vision encoder (`gemma4v`).
    Gemma4Vision,
}

impl Projector {
    /// The projector of an `mmproj` header; `None` for a text model.
    pub fn of(header: &GgufHeader) -> Result<Option<Self>> {
        if !header.is_projector() {
            return Ok(None);
        }
        // Converters write the projector at either key.
        let projector = header
            .string("clip.projector_type")
            .or_else(|| header.string("clip.vision.projector_type"));
        match projector {
            Some("qwen3vl_merger") => Ok(Some(Self::Qwen3VlMerger)),
            Some("gemma4v") => Ok(Some(Self::Gemma4Vision)),
            other => Err(ModelsError::InvalidGguf(format!(
                "mmproj projector {other:?} has no vision import yet"
            ))),
        }
    }

    /// The checkpoint name of an `mmproj` tensor; `None` for tensors the
    /// import writes itself or does not use.
    #[must_use]
    pub fn tensor_name(self, gguf: &str) -> Option<String> {
        match self {
            Self::Qwen3VlMerger => qwen::tensor_name(gguf),
            Self::Gemma4Vision => gemma::tensor_name(gguf),
        }
    }
}

/// The vision half of an import.
#[derive(Debug, Clone)]
pub struct VisionImport {
    /// `config.json` keys beside `text_config`.
    pub root: Map<String, Value>,
    pub preprocessor_config: Value,
    /// `(file name, SafeTensors bytes)` of tensors the import assembles.
    pub assembled: Option<(String, Vec<u8>)>,
}

impl VisionImport {
    /// The vision import of `mmproj` for the text model `text`.
    pub fn derive(mmproj: &GgufModel, text: &GgufModel, text_config: &Value) -> Result<Self> {
        let header = mmproj.header();
        let projector = Projector::of(header)?.ok_or_else(|| {
            ModelsError::InvalidGguf("the vision file is not an mmproj (clip) GGUF".into())
        })?;
        let [(path, _)] = mmproj.shards.as_slice() else {
            return Err(ModelsError::InvalidGguf("split mmproj files are not supported".into()));
        };
        match projector {
            Projector::Qwen3VlMerger => qwen::derive(header, path, text.header(), text_config),
            Projector::Gemma4Vision => gemma::derive(header, path, text.header(), text_config),
        }
    }
}

/// The payload bytes of a dense float tensor of `file`, which an import
/// rearranges by element and writes as `SafeTensors` under its own dtype:
/// a quantized tensor is refused, as its blocks are no elements.
fn payload(header: &GgufHeader, file: &Path, name: &str) -> Result<(Vec<usize>, String, Vec<u8>)> {
    use std::io::{Read, Seek, SeekFrom};
    let missing = || ModelsError::InvalidGguf(format!("mmproj tensor {name} is missing"));
    let stored = header.tensors.iter().find(|tensor| tensor.name == name).ok_or_else(missing)?;
    if !matches!(stored.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return Err(ModelsError::InvalidGguf(format!(
            "mmproj tensor {name} is {}, not F32, F16 or BF16, and cannot be assembled",
            stored.ggml_type
        )));
    }
    let catalog = header.catalog(file)?;
    let tensor = catalog.get(name).ok_or_else(missing)?;
    let mut bytes = vec![0; tensor.payload_bytes()?];
    let mut handle = std::fs::File::open(file)?;
    let _ = handle.seek(SeekFrom::Start(tensor.payload_start()?))?;
    handle.read_exact(&mut bytes)?;
    Ok((tensor.shape.clone(), tensor.dtype.clone(), bytes))
}

/// The id of a special token in the text model's vocabulary.
fn token_id(text: &GgufHeader, token: &str) -> Result<usize> {
    text.get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_array)
        .and_then(|tokens| tokens.iter().position(|value| value.as_str() == Some(token)))
        .ok_or_else(|| ModelsError::InvalidGguf(format!("the vocabulary has no {token}")))
}
