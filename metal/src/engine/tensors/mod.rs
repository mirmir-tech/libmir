mod gguf;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use models::{layout::WeightContainer, weights::TensorInfo};

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
        let tensors = gguf::shards(path)?.remove(path).unwrap_or_default();
        Self::load_gguf_shard(path, &tensors, full_precision, stream)
    }

    /// One shard of a GGUF model: its `tensors`, its share of the model's
    /// mapped catalog ([`GgufShards`]), keeping `full_precision` as
    /// [`Self::load_keeping`] does.
    pub fn load_gguf_shard(
        path: &Path,
        tensors: &[TensorInfo],
        full_precision: &BTreeSet<String>,
        stream: &Stream,
    ) -> Result<Self> {
        let ranges = gguf::ranges(tensors)?;
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

/// The tensors of each shard of a model's GGUF weights, from one mapped
/// catalog built for the whole model; empty for other weights.
#[derive(Debug, Default)]
pub struct GgufShards(BTreeMap<PathBuf, Vec<TensorInfo>>);

impl GgufShards {
    /// Maps the model of the first GGUF file in `paths`, once.
    pub fn of<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Result<Self> {
        paths
            .into_iter()
            .find(|path| WeightContainer::of(path) == Some(WeightContainer::Gguf))
            .map_or_else(|| Ok(Self::default()), |path| Ok(Self(gguf::shards(path)?)))
    }

    /// Loads `path`: a shard by its share of the catalog, other weights
    /// whole.
    pub fn load(
        &mut self,
        path: &Path,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
    ) -> Result<TensorFile> {
        if WeightContainer::of(path) != Some(WeightContainer::Gguf) {
            return TensorFile::load_keeping(path, full_precision, stream);
        }
        let tensors = self.0.remove(path).ok_or_else(|| {
            super::Error::InvalidModel(format!(
                "{} is not a shard of the model's first GGUF file",
                path.display()
            ))
        })?;
        TensorFile::load_gguf_shard(path, &tensors, full_precision, stream)
    }
}
