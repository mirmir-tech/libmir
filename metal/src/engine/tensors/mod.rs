mod gguf;
#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, PoisonError},
};

use models::{
    layout::{WeightContainer, WeightFile},
    weights::TensorInfo,
};

use super::{Array, Dtype, Error, Result, Stream};

#[derive(Debug)]
pub struct TensorFile {
    /// Behind a lock so a loader that copies a tensor into another (an
    /// expert bank) can take it out through a shared reference.
    native: Mutex<mirtal::TensorFile>,
    /// GGUF float tensors cast to the BF16 activation type, by name, taken
    /// out of `native`: the cast is the only owner of the stored array, which
    /// is freed once the cast has run, so a tensor occupies memory once.
    converted: Mutex<BTreeMap<String, mirtal::Array>>,
}

impl TensorFile {
    /// Binds a safetensors file. A GGUF file loads through [`GgufShards`],
    /// by its share of its model's catalog; nothing is read for it here.
    pub fn load(path: &Path, stream: &Stream) -> Result<Self> {
        if WeightContainer::of(path) == Some(WeightContainer::Gguf) {
            return Err(Error::UncataloguedGguf(path.to_owned()));
        }
        Ok(Self {
            native: Mutex::new(mirtal::TensorFile::load(path, stream.native())?),
            converted: Mutex::default(),
        })
    }

    /// One file of a GGUF model: its `tensors`, its share of the model's
    /// mapped catalog ([`GgufShards`]), keeping the float tensors named in
    /// `full_precision` in their stored type: the binding plan's
    /// [`models::weights::TensorBinding::full_precision_sources`].
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
        Ok(Self {
            native: Mutex::new(native),
            converted: Mutex::new(converted),
        })
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.native().len() + self.converted().len())
    }

    /// Evaluates every tensor; the casts one at a time, so a stored tensor is
    /// freed before the next is read.
    pub fn evaluate(&self, stream: &Stream) -> Result<()> {
        self.native().evaluate()?;
        for value in self.converted().values() {
            stream.native().eval(value)?;
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        if let Some(value) = self.converted().get(name) {
            return Array::from_native(value.clone());
        }
        Array::from_native(self.native().get(name)?)
    }

    /// Removes `name` and returns it, so the file no longer keeps what it
    /// reads alive: a loader that copies it into another array frees it once
    /// the copy is evaluated.
    pub fn take(&self, name: &str) -> Result<Array> {
        let converted = self.converted().remove(name);
        match converted {
            Some(value) => Array::from_native(value),
            None => Array::from_native(self.native().take(name)?),
        }
    }

    pub fn contains(&self, name: &str) -> Result<bool> {
        Ok(self.converted().contains_key(name) || self.native().contains(name)?)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.native().is_empty() && self.converted().is_empty())
    }

    fn native(&self) -> MutexGuard<'_, mirtal::TensorFile> {
        self.native.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn converted(&self) -> MutexGuard<'_, BTreeMap<String, mirtal::Array>> {
        self.converted.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The tensors of each GGUF file of a model's weights, from one mapped
/// catalog per GGUF group: a text model and a separate projector are
/// groups of their own, each mapped once across its shards. Empty for
/// other weights.
#[derive(Debug, Default)]
pub struct GgufShards(BTreeMap<PathBuf, Vec<TensorInfo>>);

impl GgufShards {
    /// Maps every GGUF group among `weights`, whatever order they sort in.
    pub fn of(weights: &[WeightFile]) -> Result<Self> {
        let gguf = weights
            .iter()
            .filter(|weight| weight.container == WeightContainer::Gguf)
            .cloned()
            .collect::<Vec<_>>();
        let mut files = gguf
            .iter()
            .map(|weight| (weight.path.clone(), Vec::new()))
            .collect::<BTreeMap<_, _>>();
        for tensor in WeightContainer::Gguf.read_catalog(&gguf)? {
            files.entry(tensor.file.clone()).or_default().push(tensor);
        }
        Ok(Self(files))
    }

    /// Loads `path`: a GGUF file by its share of its group's catalog, keeping
    /// `full_precision`; other weights whole.
    pub fn load(
        &mut self,
        path: &Path,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
    ) -> Result<TensorFile> {
        if WeightContainer::of(path) != Some(WeightContainer::Gguf) {
            return TensorFile::load(path, stream);
        }
        let tensors =
            self.0.remove(path).ok_or_else(|| Error::UncataloguedGguf(path.to_owned()))?;
        TensorFile::load_gguf_shard(path, &tensors, full_precision, stream)
    }
}
