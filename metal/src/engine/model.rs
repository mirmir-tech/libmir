use std::path::Path;

use models::{layout::ModelLayout, weights::alternate_model_tensor_name};

use super::{Array, Error, Result, Stream, TensorFile};

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
        Self::load_layout_inner(layout, stream, false, progress)
    }

    pub fn load_layout_materialized_with_progress(
        layout: &ModelLayout,
        stream: &Stream,
        progress: impl FnMut(u64, u64, String),
    ) -> Result<Self> {
        Self::load_layout_inner(layout, stream, true, progress)
    }

    fn load_layout_inner(
        layout: &ModelLayout,
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
        for (index, weight) in layout.weights.iter().enumerate() {
            progress(loaded, total, shard_detail(starting, index, shards, &weight.path));
            let file = TensorFile::load(&weight.path, stream)?;
            tensors += file.len()?;
            if materialize {
                file.evaluate()?;
            }
            files.push(file);
            loaded = loaded.saturating_add(weight.bytes).min(total);
            progress(loaded, total, shard_detail(complete, index, shards, &weight.path));
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

    pub fn evaluate(&self) -> Result<()> {
        self.files.iter().try_for_each(TensorFile::evaluate)
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn reports_a_shard_only_after_materialization() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "libmir-materialized-progress-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&root)?;
        fs::write(root.join("config.json"), "{}")?;
        write_safetensors(&root.join("model.safetensors"))?;
        let layout = ModelLayout::inspect(&root)?;
        let mut events = Vec::new();

        let tensors = ModelTensors::load_layout_materialized_with_progress(
            &layout,
            &Stream::new_cpu()?,
            |current, total, detail| events.push((current, total, detail)),
        )?;

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, 0);
        assert!(events[0].2.starts_with("materializing weight shard 1/1"));
        assert_eq!(events[1].0, events[1].1);
        assert!(events[1].2.starts_with("materialized weight shard 1/1"));
        drop(tensors);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn write_safetensors(path: &Path) -> Result<()> {
        let mut header =
            r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_owned();
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(&1.0_f32.to_le_bytes());
        fs::write(path, data)?;
        Ok(())
    }
}
