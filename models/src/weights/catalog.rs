use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};

use serde::Deserialize;

use crate::{
    error::Result,
    layout::{ModelLayout, WeightFile},
};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub file: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_start: u64,
    pub data_offsets: [u64; 2],
}

impl TensorInfo {
    pub fn payload_start(&self) -> Result<u64> {
        self.data_start
            .checked_add(self.data_offsets[0])
            .ok_or_else(|| crate::ModelsError::InvalidTensorRange(self.name.clone()))
    }

    pub fn payload_bytes(&self) -> Result<usize> {
        let bytes = self.data_offsets[1]
            .checked_sub(self.data_offsets[0])
            .ok_or_else(|| crate::ModelsError::InvalidTensorRange(self.name.clone()))?;
        Ok(usize::try_from(bytes)?)
    }
}

#[derive(Debug, Clone)]
pub struct TensorCatalog {
    pub tensors: Vec<TensorInfo>,
}

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl TensorCatalog {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let mut tensors = Vec::new();
        for weight in &layout.weights {
            tensors.extend(read_weight_header(weight)?);
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Self { tensors })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.iter().any(|tensor| tensor.name == name)
    }

    pub fn by_prefix<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a TensorInfo> + 'a {
        self.tensors.iter().filter(move |tensor| tensor.name.starts_with(prefix))
    }
}

fn read_weight_header(weight: &WeightFile) -> Result<Vec<TensorInfo>> {
    let mut file = File::open(&weight.path)?;
    let header_len = read_header_len(&mut file)?;
    let mut header = vec![0; usize::try_from(header_len)?];
    file.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header)?;
    let metadata: BTreeMap<String, serde_json::Value> = serde_json::from_str(header)?;

    let data_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| crate::ModelsError::InvalidTensorRange(weight.path.display().to_string()))?;
    metadata
        .into_iter()
        .filter(|(name, _value)| name != "__metadata__")
        .map(|(name, value)| {
            let tensor: HeaderTensor = serde_json::from_value(value)?;
            Ok(TensorInfo {
                name,
                file: weight.path.clone(),
                dtype: tensor.dtype,
                shape: tensor.shape,
                data_start,
                data_offsets: tensor.data_offsets,
            })
        })
        .collect()
}

fn read_header_len(file: &mut File) -> Result<u64> {
    let mut bytes = [0; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    #[test]
    fn reads_safetensors_header_without_tensor_payload() -> Result<()> {
        let header =
            r#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let path = temp_path();
        let mut bytes = Vec::new();
        let header_len = u64::try_from(header.len())?;
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0; 8]);
        fs::write(&path, &bytes)?;

        let weight = WeightFile {
            path: path.clone(),
            bytes: u64::try_from(bytes.len())?,
        };
        let tensors = read_weight_header(&weight)?;
        let _removed = fs::remove_file(path);

        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(tensors[0].shape, vec![2, 2]);
        assert_eq!(tensors[0].payload_start()?, 8 + header_len);
        assert_eq!(tensors[0].payload_bytes()?, 8);
        Ok(())
    }

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!("safetensors-header-{}.safetensors", std::process::id()))
    }
}
