//! The small F32 vectors a GGUF model's configuration derives from: Llama 3
//! and Gemma 4's `rope_freqs` and Phi-3's `rope_factors_*`. An opened model
//! reads them from its files; a Hub variant checked before its download
//! carries the bytes fetched beside its headers.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Range,
};

use super::{GgmlType, GgufHeader, GgufModel, GgufTensor};
use crate::error::{ModelsError, Result};

/// The most bytes a configuration vector holds: a value per rotary pair or
/// frequency is a few kilobytes.
const MAX_VECTOR_BYTES: u64 = 1024 * 1024;

/// A vector the configuration derives from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigurationVector {
    /// Per-pair `RoPE` divisors (Llama 3) or rotated-pair marks (Gemma 4).
    RopeFrequencies,
    /// Phi-3 `LongRoPE` factors beyond the original context.
    LongRopeFactors,
    /// Phi-3 `LongRoPE` factors within the original context.
    ShortRopeFactors,
}

impl ConfigurationVector {
    pub const ALL: [Self; 3] =
        [Self::RopeFrequencies, Self::LongRopeFactors, Self::ShortRopeFactors];

    /// The GGUF tensor name.
    #[must_use]
    pub const fn tensor(self) -> &'static str {
        match self {
            Self::RopeFrequencies => "rope_freqs.weight",
            Self::LongRopeFactors => "rope_factors_long.weight",
            Self::ShortRopeFactors => "rope_factors_short.weight",
        }
    }

    fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|vector| vector.tensor() == name)
    }
}

/// Where a model's configuration vectors are read.
#[derive(Debug, Clone, Default)]
pub(super) enum VectorSource {
    /// From the shard files.
    #[default]
    Files,
    /// Bytes fetched with the headers, by vector.
    Fetched(BTreeMap<String, Vec<u8>>),
}

impl GgufHeader {
    /// The configuration vectors this header's file holds, and the byte
    /// range of each in the file.
    pub fn configuration_vectors(
        &self,
    ) -> impl Iterator<Item = (ConfigurationVector, Range<u64>)> + '_ {
        self.tensors.iter().filter_map(|tensor| {
            let vector = ConfigurationVector::named(&tensor.name)?;
            let start = self.data_start.checked_add(tensor.offset)?;
            Some((vector, start..start.checked_add(tensor.bytes)?))
        })
    }
}

impl GgufModel {
    /// A model of `shards`' headers alone, checked before its files
    /// download, with the bytes of the configuration vectors their
    /// [`GgufHeader::configuration_vectors`] name, fetched from the files.
    pub fn from_headers(
        shards: Vec<(std::path::PathBuf, GgufHeader)>,
        vectors: impl IntoIterator<Item = (ConfigurationVector, Vec<u8>)>,
    ) -> Result<Self> {
        if shards.is_empty() {
            return Err(ModelsError::InvalidGguf("a GGUF model needs one header".into()));
        }
        let vectors = vectors
            .into_iter()
            .map(|(vector, bytes)| (vector.tensor().to_owned(), bytes))
            .collect();
        let model = Self {
            shards,
            vectors: VectorSource::Fetched(vectors),
        };
        model.validate()?;
        Ok(model)
    }

    /// The configuration vector's values; `None` when no shard holds it.
    pub(super) fn vector(&self, vector: ConfigurationVector) -> Option<Result<Vec<f64>>> {
        let name = vector.tensor();
        let (path, header, tensor) = self.shards.iter().find_map(|(path, header)| {
            Some((path, header, header.tensors.iter().find(|tensor| tensor.name == name)?))
        })?;
        Some(
            check(tensor)
                .and_then(|()| self.vector_bytes(path, header, tensor))
                .and_then(|bytes| decode(tensor, &bytes)),
        )
    }

    fn vector_bytes(
        &self,
        path: &std::path::Path,
        header: &GgufHeader,
        tensor: &GgufTensor,
    ) -> Result<Vec<u8>> {
        match &self.vectors {
            VectorSource::Files => {
                let mut bytes = vec![0_u8; usize::try_from(tensor.bytes)?];
                let mut file = File::open(path)?;
                file.seek(SeekFrom::Start(header.data_start + tensor.offset))?;
                file.read_exact(&mut bytes)?;
                Ok(bytes)
            },
            VectorSource::Fetched(vectors) => vectors.get(&tensor.name).cloned().ok_or_else(|| {
                ModelsError::InvalidGguf(format!(
                    "{} was not fetched with the headers",
                    tensor.name
                ))
            }),
        }
    }
}

/// Refuses a vector the configuration cannot be derived from before any
/// of it is read: one that is not a small vector of F32.
fn check(tensor: &GgufTensor) -> Result<()> {
    if tensor.ggml_type != GgmlType::F32 || tensor.dims.len() != 1 {
        return Err(ModelsError::InvalidGguf(format!("{} is not a vector of F32", tensor.name)));
    }
    if tensor.bytes > MAX_VECTOR_BYTES {
        return Err(ModelsError::InvalidGguf(format!(
            "{} holds {} bytes, beyond {MAX_VECTOR_BYTES}",
            tensor.name, tensor.bytes
        )));
    }
    Ok(())
}

/// The vector's values, every one finite: a configuration derived from a
/// `NaN` or infinity would pass checks that compare against it.
fn decode(tensor: &GgufTensor, bytes: &[u8]) -> Result<Vec<f64>> {
    if u64::try_from(bytes.len())? != tensor.bytes {
        return Err(ModelsError::InvalidGguf(format!("{} has the wrong length", tensor.name)));
    }
    let (words, _) = bytes.as_chunks::<4>();
    let values = words
        .iter()
        .map(|word| f64::from(f32::from_le_bytes(*word)))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ModelsError::InvalidGguf(format!("{} holds a non-finite value", tensor.name)));
    }
    Ok(values)
}
