use crate::engine::{Array, DenseLinear, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct VisionMlp {
    gate: DenseLinear,
    up: DenseLinear,
    down: DenseLinear,
}

impl VisionMlp {
    pub(super) fn load(tensors: &ModelTensors, prefix: &str, stream: &Stream) -> Result<Self> {
        Ok(Self {
            gate: DenseLinear::load_clippable(tensors, &format!("{prefix}.gate_proj"), stream)?,
            up: DenseLinear::load_clippable(tensors, &format!("{prefix}.up_proj"), stream)?,
            down: DenseLinear::load_clippable(tensors, &format!("{prefix}.down_proj"), stream)?,
        })
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let gate = self.gate.forward(input, stream)?.gelu_tanh(stream)?;
        let up = self.up.forward(input, stream)?;
        self.down.forward(&gate.multiply(&up, stream)?, stream)
    }
}
