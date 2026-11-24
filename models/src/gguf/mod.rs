//! GGUF container support for versions 2 and 3.
//!
//! A bounds-checked parser reads the header, metadata and tensor directory and
//! builds the tensor catalog the binding layer reads. Payloads are never read.

mod config;
mod dequantize;
mod fused;
mod header;
mod import;
mod names;
mod reader;
mod split;
mod tensor;
#[cfg(test)]
mod tests;
mod tiled;
mod tokenizer;
mod types;
mod value;
mod vision;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub use config::imports_architecture;
pub use dequantize::dequantize;
pub use header::GgufHeader;
pub use import::Import;
pub use names::safetensors_name;
pub use reader::ParseError;
pub use split::{GgufModel, shard_paths};
pub use tensor::GgufTensor;
pub use tiled::TiledValueHeads;
pub use types::{Affine32, GgmlType};
pub use value::MetadataValue;
pub use vision::{Projector, VisionImport};

use crate::{error::Result, layout::WeightContainer};

pub const MAGIC: [u8; 4] = *b"GGUF";

/// Payload bytes per GGML type, largest first.
fn type_bytes<'a>(tensors: impl IntoIterator<Item = &'a GgufTensor>) -> Vec<(GgmlType, u64)> {
    let mut totals = BTreeMap::new();
    for tensor in tensors {
        *totals.entry(tensor.ggml_type).or_insert(0_u64) += tensor.bytes;
    }
    let mut totals: Vec<_> = totals.into_iter().collect();
    totals.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    totals
}

/// Whether a path names a GGUF file.
#[must_use]
pub fn is_gguf(path: &Path) -> bool {
    WeightContainer::of(path) == Some(WeightContainer::Gguf)
}

/// The GGUF files directly inside a directory, sorted.
pub fn files_in(directory: &Path) -> Result<Vec<PathBuf>> {
    WeightContainer::Gguf.discover(directory)
}

/// The text model among the GGUF files in a directory, opened with all its
/// shards: an import links a vision tower's `mmproj` file
/// (`general.architecture = clip`) beside it, which may sort first.
pub fn text_model(directory: &Path) -> Result<Option<GgufModel>> {
    for path in files_in(directory)? {
        let model = GgufModel::open(&path)?;
        if model.header().architecture() != Some("clip") {
            return Ok(Some(model));
        }
    }
    Ok(None)
}
