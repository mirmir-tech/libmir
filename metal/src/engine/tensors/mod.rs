mod gguf;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use super::{Array, Dtype, Result, Stream};

#[derive(Debug)]
pub struct TensorFile {
    native: mirtal::TensorFile,
    /// GGUF float tensors cast to the BF16 activation type, by name. The cast
    /// is lazy: it runs once, on first use, after the bytes are read.
    converted: BTreeMap<String, mirtal::Array>,
}

impl TensorFile {
    /// Binds a safetensors file, or a GGUF file whose tensors are read in
    /// place under their mapped `SafeTensors` names.
    pub fn load(path: &Path, stream: &Stream) -> Result<Self> {
        Self::load_keeping(path, &BTreeSet::new(), stream)
    }

    /// [`Self::load`], keeping the GGUF float tensors named in
    /// `full_precision` in their stored type: the binding plan's
    /// [`models::weights::TensorBinding::full_precision_sources`].
    pub fn load_keeping(
        path: &Path,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
    ) -> Result<Self> {
        if !models::gguf::is_gguf(path) {
            return Ok(Self {
                native: mirtal::TensorFile::load(path, stream.native())?,
                converted: BTreeMap::new(),
            });
        }
        let ranges = gguf::ranges(path)?;
        let native = mirtal::TensorFile::load_ranges(path, &ranges, stream.native())?;
        let mut converted = BTreeMap::new();
        for range in ranges.iter().filter(|range| gguf::casts_to_bf16(range, full_precision)) {
            let value = Array::from_native(native.get(&range.name)?)?;
            let value = value.astype(Dtype::Bfloat16, stream)?;
            converted.insert(range.name.clone(), value.native().clone());
        }
        Ok(Self { native, converted })
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.native.len())
    }

    pub fn evaluate(&self) -> Result<()> {
        Ok(self.native.evaluate()?)
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        match self.converted.get(name) {
            Some(value) => Array::from_native(value.clone()),
            None => Array::from_native(self.native.get(name)?),
        }
    }

    pub fn contains(&self, name: &str) -> Result<bool> {
        Ok(self.native.contains(name)?)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.native.is_empty())
    }
}
