mod gguf;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use models::layout::WeightContainer;

use super::{Array, Dtype, Result, Stream};

#[derive(Debug)]
pub struct TensorFile {
    native: mirtal::TensorFile,
    /// GGUF float tensors cast to the BF16 activation type, by name, taken
    /// out of `native`: the cast is the only owner of the stored array, which
    /// is freed once the cast has run, so a tensor occupies memory once.
    converted: BTreeMap<String, mirtal::Array>,
}

impl TensorFile {
    /// Binds a safetensors file, or a GGUF file whose tensors are read from
    /// their ranges under their mapped `SafeTensors` names.
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
        if WeightContainer::of(path) != Some(WeightContainer::Gguf) {
            return Ok(Self {
                native: mirtal::TensorFile::load(path, stream.native())?,
                converted: BTreeMap::new(),
            });
        }
        let ranges = gguf::ranges(path)?;
        let mut native = mirtal::TensorFile::load_ranges(path, &ranges, stream.native())?;
        let mut converted = BTreeMap::new();
        for range in ranges.iter().filter(|range| gguf::casts_to_bf16(range, full_precision)) {
            let value = Array::from_native(native.take(range.name())?)?;
            let value = value.astype(Dtype::Bfloat16, stream)?;
            converted.insert(range.name().to_owned(), value.native().clone());
        }
        Ok(Self { native, converted })
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.native.len() + self.converted.len())
    }

    /// Evaluates every tensor; the casts one at a time, so a stored tensor is
    /// freed before the next is read.
    pub fn evaluate(&self, stream: &Stream) -> Result<()> {
        self.native.evaluate()?;
        for value in self.converted.values() {
            stream.native().eval(value)?;
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        match self.converted.get(name) {
            Some(value) => Array::from_native(value.clone()),
            None => Array::from_native(self.native.get(name)?),
        }
    }

    pub fn contains(&self, name: &str) -> Result<bool> {
        Ok(self.converted.contains_key(name) || self.native.contains(name)?)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.native.is_empty() && self.converted.is_empty())
    }
}
