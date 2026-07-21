use crate::engine::{Array, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct DenseEmbedding {
    weight: Array,
}

impl DenseEmbedding {
    pub fn load(tensors: &ModelTensors, prefix: &str) -> Result<Self> {
        Self::load_name(tensors, &format!("{prefix}.weight"))
    }

    pub(in crate::engine) fn load_name(tensors: &ModelTensors, weight: &str) -> Result<Self> {
        Ok(Self { weight: tensors.get(weight)? })
    }

    pub fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        self.weight.take(indices, 0, stream)
    }
}
