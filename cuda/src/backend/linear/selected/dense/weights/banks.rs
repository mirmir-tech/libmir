use super::{DenseExpertWeights, DenseGateUpWeights, DenseProjectionWeight};
use crate::CudaTensor;

impl DenseExpertWeights {
    /// Canonical BF16 banks without biases: gate/up `[experts, 2 * out, in]`
    /// with gate row `2j` and up row `2j + 1`, and down; such as GGML banks
    /// dequantized into scratch.
    pub(in crate::backend) fn from_interleaved_banks(
        gate_up: CudaTensor,
        down: CudaTensor,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Self {
        let plain = |weight| DenseProjectionWeight { weight, bias: None, transposed: false };
        Self {
            gate_up: DenseGateUpWeights::Fused {
                projection: plain(gate_up),
                interleaved: true,
            },
            down: plain(down),
            experts,
            hidden,
            intermediate,
        }
    }
}
