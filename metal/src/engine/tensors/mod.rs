mod gguf;
mod inline;
#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, PoisonError},
};

use models::{
    layout::{WeightContainer, WeightFile},
    weights::{TensorCatalog, TensorInfo},
};

use super::{Array, Dtype, Error, Result, Stream};

#[derive(Debug)]
pub struct TensorFile {
    /// Behind a lock so a loader that copies a tensor into another (an
    /// expert bank) can take it out through a shared reference. `None` for
    /// in-memory `SafeTensors` payloads (assembled GGUF vision projections).
    native: Option<Mutex<mirtal::TensorFile>>,
    /// GGUF float tensors cast to the BF16 activation type (or in-memory
    /// tensors), by name.
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
            native: Some(Mutex::new(mirtal::TensorFile::load(path, stream.native())?)),
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
            native: Some(Mutex::new(native)),
            converted: Mutex::new(converted),
        })
    }

    pub fn len(&self) -> Result<usize> {
        let native = self.native().map_or(0, |guard| guard.len());
        let converted = self.converted().len();
        Ok(native + converted)
    }

    /// Evaluates every tensor; the casts one at a time, so a stored tensor is
    /// freed before the next is read.
    pub fn evaluate(&self, stream: &Stream) -> Result<()> {
        if let Some(native) = self.native() {
            native.evaluate()?;
        }
        for value in self.converted().values() {
            stream.native().eval(value)?;
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        if let Some(value) = self.converted().get(name) {
            return Array::from_native(value.clone());
        }
        let native = self.native().ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        Array::from_native(native.get(name)?)
    }

    /// Removes `name` and returns it, so the file no longer keeps what it
    /// reads alive: a loader that copies it into another array frees it once
    /// the copy is evaluated.
    pub fn take(&self, name: &str) -> Result<Array> {
        let converted = self.converted().remove(name);
        if let Some(value) = converted {
            Array::from_native(value)
        } else {
            let mut native = self.native().ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
            Array::from_native(native.take(name)?)
        }
    }

    pub fn contains(&self, name: &str) -> Result<bool> {
        if self.converted().contains_key(name) {
            return Ok(true);
        }
        self.native().map_or(Ok(false), |native| Ok(native.contains(name)?))
    }

    pub fn is_empty(&self) -> Result<bool> {
        let native = self.native().is_none_or(|guard| guard.is_empty());
        let converted = self.converted().is_empty();
        Ok(native && converted)
    }

    fn native(&self) -> Option<MutexGuard<'_, mirtal::TensorFile>> {
        self.native
            .as_ref()
            .map(|mutex| mutex.lock().unwrap_or_else(PoisonError::into_inner))
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
    /// Groups the GGUF tensors of an already-parsed catalog by shard path so
    /// loading never re-reads GGUF headers.
    #[must_use]
    pub fn from_catalog(weights: &[WeightFile], catalog: &TensorCatalog) -> Self {
        let mut files = weights
            .iter()
            .filter(|weight| weight.container == WeightContainer::Gguf)
            .map(|weight| (weight.path.clone(), Vec::new()))
            .collect::<BTreeMap<_, _>>();
        for tensor in &catalog.tensors {
            if tensor.container == WeightContainer::Gguf {
                files.entry(tensor.file.clone()).or_default().push(tensor.clone());
            }
        }
        Self(files)
    }

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

    /// Loads `weight`: a GGUF file by its share of its group's catalog, keeping
    /// `full_precision`; `SafeTensors` files whole.
    pub fn load_weight(
        &mut self,
        weight: &WeightFile,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
    ) -> Result<TensorFile> {
        match weight.container {
            WeightContainer::SafeTensors => TensorFile::load(&weight.path, stream),
            WeightContainer::Gguf => {
                let tensors = self
                    .0
                    .remove(&weight.path)
                    .ok_or_else(|| Error::UncataloguedGguf(weight.path.clone()))?;
                TensorFile::load_gguf_shard(&weight.path, &tensors, full_precision, stream)
            },
        }
    }
}
