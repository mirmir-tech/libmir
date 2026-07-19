use crate::engine::{Array, DenseLinear, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct VisionMlp {
    input: DenseLinear,
    output: DenseLinear,
}

impl VisionMlp {
    pub(super) fn load(tensors: &ModelTensors, prefix: &str, stream: &Stream) -> Result<Self> {
        Ok(Self {
            input: DenseLinear::load(tensors, &format!("{prefix}.linear_fc1"), stream)?,
            output: DenseLinear::load(tensors, &format!("{prefix}.linear_fc2"), stream)?,
        })
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        self.output
            .forward(&self.input.forward(input, stream)?.gelu_tanh(stream)?, stream)
    }
}
