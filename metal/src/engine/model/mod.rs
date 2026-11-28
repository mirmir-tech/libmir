#[cfg(test)]
mod tests;

use std::{collections::BTreeSet, path::Path};

use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, alternate_model_tensor_name},
};

use super::{Array, Error, Result, Stream, TensorFile, tensors::GgufShards};

#[derive(Debug)]
pub struct ModelTensors {
    files: Vec<TensorFile>,
    tensors: usize,
}

impl ModelTensors {
    pub fn load(root: impl AsRef<Path>, stream: &Stream) -> Result<Self> {
        let layout = ModelLayout::inspect(root)?;
        Self::load_layout(&layout, stream)
    }

    pub fn load_layout(layout: &ModelLayout, stream: &Stream) -> Result<Self> {
        Self::load_layout_with_progress(layout, stream, |_loaded, _total, _detail| {})
    }

    pub fn load_layout_with_progress(
        layout: &ModelLayout,
        stream: &Stream,
        progress: impl FnMut(u64, u64, String),
    ) -> Result<Self> {
        Self::load_layout_inner(layout, None, &BTreeSet::new(), stream, false, progress)
    }

    pub fn load_layout_materialized_with_progress(
        layout: &ModelLayout,
        stream: &Stream,
        progress: impl FnMut(u64, u64, String),
    ) -> Result<Self> {
        Self::load_layout_inner(layout, None, &BTreeSet::new(), stream, true, progress)
    }

    /// [`Self::load_layout_materialized_with_progress`] reusing an already
    /// parsed catalog and keeping `full_precision` GGUF float tensors in their
    /// stored type.
    pub fn load_catalog_materialized_keeping(
        layout: &ModelLayout,
        catalog: &TensorCatalog,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
        progress: impl FnMut(u64, u64, String),
    ) -> Result<Self> {
        Self::load_layout_inner(layout, Some(catalog), full_precision, stream, true, progress)
    }

    /// [`Self::load_layout_materialized_with_progress`] keeping the GGUF
    /// float tensors in `full_precision` in their stored type.
    pub fn load_layout_materialized_keeping(
        layout: &ModelLayout,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
        progress: impl FnMut(u64, u64, String),
    ) -> Result<Self> {
        Self::load_layout_inner(layout, None, full_precision, stream, true, progress)
    }

    fn load_layout_inner(
        layout: &ModelLayout,
        catalog: Option<&TensorCatalog>,
        full_precision: &BTreeSet<String>,
        stream: &Stream,
        materialize: bool,
        mut progress: impl FnMut(u64, u64, String),
    ) -> Result<Self> {
        let mut files = Vec::with_capacity(layout.weights.len());
        let mut tensors = 0;
        let total = layout
            .weights
            .iter()
            .fold(0_u64, |total, weight| total.saturating_add(weight.bytes));
        let mut loaded = 0;
        let shards = layout.weights.len();
        let (starting, complete) = if materialize {
            ("materializing", "materialized")
        } else {
            ("mapping", "mapped")
        };
        let mut gguf = match catalog {
            Some(catalog) => GgufShards::from_catalog(&layout.weights, catalog),
            None => GgufShards::of(&layout.weights)?,
        };
        for (index, weight) in layout.weights.iter().enumerate() {
            progress(loaded, total, shard_detail(starting, index, shards, &weight.path));
            let file = gguf.load_weight(weight, full_precision, stream)?;
            tensors += file.len()?;
            if materialize {
                file.evaluate(stream)?;
            }
            files.push(file);
            loaded = loaded.saturating_add(weight.bytes).min(total);
            progress(loaded, total, shard_detail(complete, index, shards, &weight.path));
        }
        if let Some((name, bytes)) = layout
            .derived
            .as_ref()
            .and_then(|derived| derived.assembled_safetensors.as_ref())
        {
            let file = TensorFile::from_safetensors_bytes(name, bytes, stream)?;
            tensors += file.len()?;
            if materialize {
                file.evaluate(stream)?;
            }
            files.push(file);
        }
        Ok(Self { files, tensors })
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        if let Some(tensor) = self.get_exact(name)? {
            return Ok(tensor);
        }
        let alternate = alternate_model_tensor_name(name);
        if let Some(tensor) = self.get_exact(&alternate)? {
            return Ok(tensor);
        }
        Err(Error::MissingTensor(name.to_owned()))
    }

    pub fn get_optional(&self, name: &str) -> Result<Option<Array>> {
        if let Some(tensor) = self.get_exact(name)? {
            return Ok(Some(tensor));
        }
        self.get_exact(&alternate_model_tensor_name(name))
    }

    /// [`Self::get`], removing the tensor so the model's files no longer
    /// keep it: for a loader that copies it into another array (an expert
    /// bank), which is then its only owner.
    pub(in crate::engine) fn take(&self, name: &str) -> Result<Array> {
        for name in [name.to_owned(), alternate_model_tensor_name(name)] {
            if let Some(file) = self.file_with(&name)? {
                return file.take(&name);
            }
        }
        Err(Error::MissingTensor(name.to_owned()))
    }

    fn file_with(&self, name: &str) -> Result<Option<&TensorFile>> {
        for file in &self.files {
            if file.contains(name)? {
                return Ok(Some(file));
            }
        }
        Ok(None)
    }

    fn get_exact(&self, name: &str) -> Result<Option<Array>> {
        for file in &self.files {
            if file.contains(name)? {
                return file.get(name).map(Some);
            }
        }
        Ok(None)
    }

    pub fn contains(&self, name: &str) -> Result<bool> {
        Ok(
            self.contains_exact(name)?
                || self.contains_exact(&alternate_model_tensor_name(name))?,
        )
    }

    fn contains_exact(&self, name: &str) -> Result<bool> {
        for file in &self.files {
            if file.contains(name)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn evaluate(&self, stream: &Stream) -> Result<()> {
        self.files.iter().try_for_each(|file| file.evaluate(stream))
    }

    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors == 0
    }
}

fn shard_detail(action: &str, index: usize, total: usize, path: &Path) -> String {
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("weights");
    format!("{action} weight shard {}/{} · {name}", index + 1, total)
}
