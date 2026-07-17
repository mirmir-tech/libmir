use mircuda::{DeviceBuffer, Stream, bf16};

use super::{AffineQuantizedTensors, u32_tensor, validate_bank};
use crate::{
    CudaBackend, Error, Result,
    backend::linear::quantized::{bf16_tensor, expected_shape},
    kernels::{
        AffineGemvSpec, SelectedAffinePair, SelectedAffinePairLaunch, SelectedAffinePairSpec,
    },
};

/// Gate and up checkpoint tensor banks sharing one affine format.
#[derive(Clone, Copy)]
pub struct AffineQuantizedPairTensors<'a> {
    pub gate: AffineQuantizedTensors<'a>,
    pub up: AffineQuantizedTensors<'a>,
}

/// Fused paired projections over device-selected experts.
#[derive(Clone, Debug)]
pub struct SelectedAffinePairBf16Linear {
    operation: SelectedAffinePair,
    stream: Stream,
}

impl SelectedAffinePairBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<Self> {
        let spec = SelectedAffinePairSpec::new(matrix, expert_count, selected_count)?;
        Ok(Self {
            operation: SelectedAffinePair::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Enqueues all selected gate/up projections in one kernel launch.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        tensors: AffineQuantizedPairTensors<'_>,
        gate_output: &mut DeviceBuffer<bf16>,
        up_output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.operation.spec();
        let matrix = spec.matrix;
        let packed = matrix.input_features / (32 / matrix.bits);
        let groups = matrix.input_features / matrix.group_size;
        let weight_shape = expected_shape(spec.expert_count, matrix.output_features, packed);
        let group_shape = expected_shape(spec.expert_count, matrix.output_features, groups);
        validate_bank(tensors.gate, &weight_shape, &group_shape)?;
        validate_bank(tensors.up, &weight_shape, &group_shape)?;
        self.operation.execute(
            &self.stream,
            &mut SelectedAffinePairLaunch {
                input,
                selected,
                gate_weight: u32_tensor(tensors.gate.weight)?,
                gate_scales: bf16_tensor(tensors.gate.scales)?,
                gate_biases: bf16_tensor(tensors.gate.biases)?,
                up_weight: u32_tensor(tensors.up.weight)?,
                up_scales: bf16_tensor(tensors.up.scales)?,
                up_biases: bf16_tensor(tensors.up.biases)?,
                gate_output,
                up_output,
            },
        )
    }

    /// Elements required by each output buffer.
    pub fn output_elements(&self) -> Result<usize> {
        let spec = self.operation.spec();
        spec.matrix.output_features.checked_mul(spec.selected_count).ok_or_else(|| {
            Error::InvalidTensorSize {
                name: "selected affine pair output".into(),
                expected: usize::MAX,
                actual: 0,
            }
        })
    }
}
