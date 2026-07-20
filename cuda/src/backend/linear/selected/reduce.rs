use mircuda::{DeviceBuffer, Stream, bf16};

use super::{AffineQuantizedTensors, u32_tensor, validate_bank};
use crate::{
    CudaBackend, Result,
    backend::linear::quantized::{bf16_tensor, expected_shape},
    kernels::{
        AffineGemvSpec, SelectedAffineReduce, SelectedAffineReduceLaunch, SelectedAffineReduceSpec,
    },
};

/// Selected down projections with deterministic router-weighted reduction.
#[derive(Clone, Debug)]
pub struct SelectedAffineReduceBf16Linear {
    operation: SelectedAffineReduce,
    stream: Stream,
}

impl SelectedAffineReduceBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<Self> {
        Self::new_batch(backend, matrix, expert_count, selected_count, 1)
    }

    pub(in crate::backend) fn new_batch(
        backend: &CudaBackend,
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
        tokens: usize,
    ) -> Result<Self> {
        let spec =
            SelectedAffineReduceSpec::new_batch(matrix, expert_count, selected_count, tokens)?;
        Ok(Self {
            operation: SelectedAffineReduce::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Enqueues all selected down projections and their router reduction.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing_weights: &DeviceBuffer<bf16>,
        tensors: AffineQuantizedTensors<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.operation.spec();
        let matrix = spec.matrix;
        let packed = matrix.input_features / (32 / matrix.bits);
        let groups = matrix.input_features / matrix.group_size;
        let weight_shape = expected_shape(spec.expert_count, matrix.output_features, packed);
        let group_shape = expected_shape(spec.expert_count, matrix.output_features, groups);
        validate_bank(tensors, &weight_shape, &group_shape)?;
        self.operation.execute(
            &self.stream,
            &mut SelectedAffineReduceLaunch {
                input,
                selected,
                routing_weights,
                weight: u32_tensor(tensors.weight)?,
                scales: bf16_tensor(tensors.scales)?,
                biases: bf16_tensor(tensors.biases)?,
                output,
            },
        )
    }

    /// Elements required by the reduced hidden-state output.
    pub fn output_elements(&self) -> Result<usize> {
        let spec = self.operation.spec();
        spec.matrix.output_features.checked_mul(spec.tokens).ok_or_else(|| {
            crate::Error::InvalidTensorSize {
                name: "selected reduced output".into(),
                expected: usize::MAX,
                actual: 0,
            }
        })
    }
}
