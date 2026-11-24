//! The file format a checkpoint's weights are stored in: the only place that
//! looks at a weight file's extension.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use super::WeightFile;
use crate::{
    error::Result,
    gguf::GgufModel,
    weights::{TensorInfo, read_weight_header},
};

/// The container of one weight file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightContainer {
    /// Hugging Face `SafeTensors` files.
    SafeTensors,
    /// llama.cpp GGUF files, a model's shards or a vision tower's `mmproj`.
    Gguf,
}

impl WeightContainer {
    /// The container a file names by its extension, in any case; `None` for
    /// a file that holds no weights.
    #[must_use]
    pub fn of(path: &Path) -> Option<Self> {
        let extension = path.extension()?;
        [Self::SafeTensors, Self::Gguf]
            .into_iter()
            .find(|container| extension.eq_ignore_ascii_case(container.extension()))
    }

    /// The file extension, without the dot.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
        }
    }

    /// The files of this container directly inside `root`, sorted.
    pub fn discover(self, root: &Path) -> Result<Vec<PathBuf>> {
        let mut files = std::fs::read_dir(root)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_file() && Self::of(path) == Some(self))
            .collect::<Vec<_>>();
        files.sort();
        Ok(files)
    }

    /// The container of a model whose weight files have these containers:
    /// its text weights', GGUF when any file is GGUF, since a GGUF import
    /// keeps its vision tower's assembled tensors as `SafeTensors`.
    pub fn of_model(containers: impl IntoIterator<Item = Self>) -> Option<Self> {
        containers.into_iter().reduce(|model, file| {
            if model == Self::Gguf {
                model
            } else {
                file
            }
        })
    }

    /// The model container of a directory libmir could not inspect: GGUF
    /// when it holds GGUF files, whose import may add a vision `SafeTensors`
    /// file beside them.
    pub fn detect(root: &Path) -> Result<Option<Self>> {
        for container in [Self::Gguf, Self::SafeTensors] {
            if !container.discover(root)?.is_empty() {
                return Ok(Some(container));
            }
        }
        Ok(None)
    }

    /// The tensors of `files`, each of this container. A split GGUF model
    /// maps its names once over every shard, however many of them `files`
    /// lists.
    pub fn read_catalog(self, files: &[WeightFile]) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::new();
        match self {
            Self::SafeTensors => {
                for file in files {
                    tensors.extend(read_weight_header(file)?);
                }
            },
            Self::Gguf => {
                let mut shards = HashSet::new();
                for file in files {
                    if shards.contains(&file.path) {
                        continue;
                    }
                    let model = GgufModel::open(&file.path)?;
                    shards.extend(model.shards.iter().map(|(path, _)| path.clone()));
                    tensors.extend(model.mapped_catalog()?.tensors);
                }
            },
        }
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests;
