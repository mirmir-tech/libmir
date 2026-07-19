use super::{Array, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    pub fn load(tensors: &ModelTensors, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: tensors.get(&format!("{prefix}.weight"))?,
            bias: tensors.get(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        input.layer_norm(&self.weight, &self.bias, self.eps, stream)
    }

    #[cfg(test)]
    pub(super) const fn from_arrays(weight: Array, bias: Array, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
}
