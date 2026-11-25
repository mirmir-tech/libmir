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
mod vectors;
mod vision;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub use config::imports_architecture;
pub use dequantize::dequantize;
pub use header::GgufHeader;
pub use import::{
    FileOrigin, Import, ImportConflict, ImportManifest, ImportedFile, MANIFEST, SourceFile,
};
pub use names::safetensors_name;
pub use reader::ParseError;
pub use split::{GgufModel, ShardName, shard_paths};
pub use tensor::GgufTensor;
pub use tiled::TiledValueHeads;
pub use types::{Affine32, GgmlType};
pub use value::MetadataValue;
pub use vectors::ConfigurationVector;
pub use vision::{Projector, VisionImport};

use crate::{error::Result, layout::WeightContainer};

pub const MAGIC: [u8; 4] = *b"GGUF";

/// `general.architecture` of a vision tower's projector (`mmproj`) file.
pub const PROJECTOR_ARCHITECTURE: &str = "clip";

/// The word projector file names carry, as a prefix (`mmproj-*.gguf`, as
/// llama.cpp names them) or anywhere (`gemma-4-E4B-it-mmproj.gguf`): the
/// only sign of one before its header is read, as in a Hub listing.
const PROJECTOR_FILE_WORD: &str = "mmproj";

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

/// The GGUF files directly inside a directory, sorted: those of its import
/// when it holds one ([`ImportManifest`]), else all of them.
pub fn files_in(directory: &Path) -> Result<Vec<PathBuf>> {
    if let Some(manifest) = ImportManifest::read(directory) {
        return Ok(manifest.gguf_files(directory));
    }
    WeightContainer::Gguf.discover(directory)
}

/// The text model among the GGUF files in a directory, opened with all its
/// shards: an import links a vision tower's `mmproj` file
/// (`general.architecture = clip`) beside it, which may sort first.
pub fn text_model(directory: &Path) -> Result<Option<GgufModel>> {
    for path in files_in(directory)? {
        let model = GgufModel::open(&path)?;
        if !model.header().is_projector() {
            return Ok(Some(model));
        }
    }
    Ok(None)
}

/// The vision tower's projector file among the GGUF files in a directory,
/// told by its header.
pub fn projector(directory: &Path) -> Result<Option<PathBuf>> {
    for path in files_in(directory)? {
        if GgufHeader::read(&path)?.is_projector() {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

/// Whether a file name, without its header, names a projector; for
/// listings whose headers are not at hand.
#[must_use]
pub fn is_projector_name(name: &str) -> bool {
    name.to_ascii_lowercase().contains(PROJECTOR_FILE_WORD)
}
