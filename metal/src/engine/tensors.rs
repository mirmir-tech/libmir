use std::path::Path;

use super::{Array, Result, Stream};

#[derive(Debug)]
pub struct TensorFile {
    native: mirtal::TensorFile,
}

impl TensorFile {
    pub fn load(path: &Path, stream: &Stream) -> Result<Self> {
        Ok(Self {
            native: mirtal::TensorFile::load(path, stream.native())?,
        })
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.native.len())
    }

    pub fn evaluate(&self) -> Result<()> {
        Ok(self.native.evaluate()?)
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        Array::from_native(self.native.get(name)?)
    }

    pub fn contains(&self, name: &str) -> Result<bool> {
        Ok(self.native.contains(name)?)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.native.is_empty())
    }
}
