mod read;

use std::{collections::BTreeMap, path::Path};

#[cfg(test)]
pub(super) use read::reads;

use super::{
    GgmlType, GgufTensor, MAGIC, MetadataValue, fused, names,
    reader::{ParseError, Parsed, Reader, invalid},
    type_bytes, value,
};
use crate::{error::Result, weights::TensorCatalog};

const DEFAULT_ALIGNMENT: u64 = 32;

/// The parsed header of one GGUF file.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    /// In directory order.
    pub tensors: Vec<GgufTensor>,
    pub alignment: u64,
    /// Absolute file offset of the tensor data section.
    pub data_start: u64,
}

impl GgufHeader {
    /// Parses a header from a prefix of the file. `file_len`, when known,
    /// rejects counts and tensors that cannot fit in the file.
    pub fn parse(bytes: &[u8], file_len: Option<u64>) -> std::result::Result<Self, ParseError> {
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(invalid("not a GGUF file: the magic is not `GGUF`"));
        }
        let version = reader.u32()?;
        if !matches!(version, 2 | 3) {
            return Err(invalid(format!("GGUF version {version} is not supported (2 and 3 are)")));
        }
        let tensor_count = reader.count("tensor", 32, file_len)?;
        let metadata_count = reader.count("metadata", 13, file_len)?;
        let metadata = read_metadata(&mut reader, metadata_count, file_len)?;
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            tensors.push(GgufTensor::read(&mut reader)?);
        }
        let alignment = alignment(&metadata)?;
        let header_end =
            u64::try_from(reader.position()).map_err(|_| invalid("header too large"))?;
        let data_start = header_end
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or_else(|| invalid("data offset overflows"))?;
        let header = Self {
            version,
            metadata,
            tensors,
            alignment,
            data_start,
        };
        header.validate_tensors(file_len)?;
        Ok(header)
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    #[must_use]
    pub fn string(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetadataValue::as_str)
    }

    #[must_use]
    pub fn unsigned(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetadataValue::as_u64)
    }

    /// `general.architecture`, the prefix of the model's hyperparameter keys.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.string("general.architecture")
    }

    /// Whether this is a vision tower's `mmproj` file rather than a text
    /// model.
    #[must_use]
    pub fn is_projector(&self) -> bool {
        self.architecture() == Some(super::PROJECTOR_ARCHITECTURE)
    }

    /// The catalog of this one file under `SafeTensors` names, which the
    /// binding layer reads; tensors without a mapped role keep their GGUF
    /// name. Split models map through [`super::GgufModel::mapped_catalog`],
    /// since only their first shard carries the metadata.
    pub fn mapped_catalog(&self, file: &Path) -> Result<TensorCatalog> {
        let mut catalog = self.catalog(file)?;
        fused::split(self, &mut catalog.tensors)?;
        names::map(self, &mut catalog.tensors)?;
        Ok(TensorCatalog::new(catalog.tensors))
    }

    pub fn catalog(&self, file: &Path) -> Result<TensorCatalog> {
        let tensors = self
            .tensors
            .iter()
            .map(|tensor| tensor.info(file, self.data_start))
            .collect::<Result<_>>()?;
        Ok(TensorCatalog::new(tensors))
    }

    /// Payload bytes per GGML type, largest first.
    #[must_use]
    pub fn type_bytes(&self) -> Vec<(GgmlType, u64)> {
        type_bytes(&self.tensors)
    }

    fn validate_tensors(&self, file_len: Option<u64>) -> Parsed<()> {
        let data_len = file_len.map(|length| length.saturating_sub(self.data_start));
        let mut ranges: Vec<&GgufTensor> = self.tensors.iter().collect();
        ranges.sort_by_key(|tensor| tensor.offset);
        let mut previous: Option<&GgufTensor> = None;
        for tensor in ranges {
            if tensor.offset % self.alignment != 0 {
                return Err(invalid(format!(
                    "tensor `{}` is not aligned to {}",
                    tensor.name, self.alignment
                )));
            }
            let end = tensor
                .offset
                .checked_add(tensor.bytes)
                .ok_or_else(|| invalid("tensor range overflows"))?;
            if data_len.is_some_and(|length| end > length) {
                return Err(invalid(format!(
                    "tensor `{}` extends past the end of the file",
                    tensor.name
                )));
            }
            if let Some(previous) = previous.filter(|previous| previous.end() > tensor.offset) {
                return Err(invalid(format!(
                    "tensors `{}` and `{}` overlap",
                    previous.name, tensor.name
                )));
            }
            previous = Some(tensor);
        }
        let mut names: Vec<&str> = self.tensors.iter().map(|tensor| tensor.name.as_str()).collect();
        names.sort_unstable();
        if let Some([name, _]) = names.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(invalid(format!("tensor `{name}` appears twice")));
        }
        Ok(())
    }
}

fn read_metadata(
    reader: &mut Reader<'_>,
    count: usize,
    file_len: Option<u64>,
) -> Parsed<BTreeMap<String, MetadataValue>> {
    let mut metadata = BTreeMap::new();
    let mut elements = value::MAX_ARRAY_ELEMENTS;
    for _ in 0..count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        let value = value::read_value(reader, kind, file_len, &mut elements)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(invalid(format!("metadata key `{key}` appears twice")));
        }
    }
    Ok(metadata)
}

fn alignment(metadata: &BTreeMap<String, MetadataValue>) -> Parsed<u64> {
    match metadata.get("general.alignment") {
        None => Ok(DEFAULT_ALIGNMENT),
        Some(MetadataValue::U32(value)) if value.is_power_of_two() => Ok(u64::from(*value)),
        Some(other) => {
            Err(invalid(format!("general.alignment {other:?} is not a power of two u32")))
        },
    }
}

#[cfg(test)]
mod tests;
