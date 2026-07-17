use mircuda::{DeviceBuffer, Stream, bf16};

use super::{AffineQuantizedConfig, AffineQuantizedTensors, CudaBackend};
use crate::{
    Error, Result,
    backend::linear::quantized::{bf16_tensor, expected_shape, validate_shape},
    kernels::{AffineQmmLaunch, AffineQmmSpec, AffineQuantizedQmm},
};

/// Fixed-token BF16-input affine Int4/Int8 prefill projection.
#[derive(Clone, Debug)]
pub struct AffineQuantizedBf16Qmm {
    operation: AffineQuantizedQmm,
    stream: Stream,
    matrices: usize,
}

impl AffineQuantizedBf16Qmm {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        config: AffineQuantizedConfig,
        matrices: usize,
    ) -> Result<Self> {
        if matrices == 0 {
            return Err(Error::InvalidMatrixIndex { index: 0, matrices });
        }
        let spec = AffineQmmSpec::new(config.spec()?, tokens)?;
        Ok(Self {
            operation: AffineQuantizedQmm::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
            matrices,
        })
    }

    /// Enqueues one tiled quantized prefill projection without allocation.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        tensors: AffineQuantizedTensors<'_>,
        output: &mut DeviceBuffer<bf16>,
        matrix_index: usize,
    ) -> Result<()> {
        if matrix_index >= self.matrices {
            return Err(Error::InvalidMatrixIndex {
                index: matrix_index,
                matrices: self.matrices,
            });
        }
        let matrix = self.operation.spec().matrix;
        let packed = matrix.input_features / (32 / matrix.bits);
        let groups = matrix.input_features / matrix.group_size;
        validate_shape(
            tensors.weight,
            expected_shape(self.matrices, matrix.output_features, packed),
        )?;
        validate_shape(
            tensors.scales,
            expected_shape(self.matrices, matrix.output_features, groups),
        )?;
        validate_shape(
            tensors.biases,
            expected_shape(self.matrices, matrix.output_features, groups),
        )?;
        let weight = tensors.weight.as_u32().ok_or_else(|| Error::DTypeMismatch {
            name: tensors.weight.name().into(),
            expected: "U32",
        })?;
        self.operation.execute(
            &self.stream,
            &mut AffineQmmLaunch {
                input,
                weight,
                scales: bf16_tensor(tensors.scales)?,
                biases: bf16_tensor(tensors.biases)?,
                output,
                matrix_index,
            },
        )
    }

    /// Number of BF16 elements required by the prefill output.
    pub fn output_elements(&self) -> Result<usize> {
        let spec = self.operation.spec();
        spec.tokens.checked_mul(spec.matrix.output_features).ok_or_else(|| {
            Error::InvalidTensorSize {
                name: "quantized prefill output".into(),
                expected: usize::MAX,
                actual: 0,
            }
        })
    }
}
