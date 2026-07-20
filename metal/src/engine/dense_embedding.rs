use super::{Array, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct DenseEmbedding {
    weight: Array,
}

impl DenseEmbedding {
    pub fn load(tensors: &ModelTensors, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: tensors.get(&format!("{prefix}.weight"))?,
        })
    }

    pub fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        self.weight.take(indices, 0, stream)
    }
}
