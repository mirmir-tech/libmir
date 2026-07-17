use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, Error, Result,
    kernels::{AffineGemvLaunch, AffineGemvSpec, AffineQuantizedGemv},
};

/// Checkpoint tensors forming one affine grouped quantized weight bank.
#[derive(Clone, Copy)]
pub struct AffineQuantizedTensors<'a> {
    pub weight: &'a CudaTensor,
    pub scales: &'a CudaTensor,
    pub biases: &'a CudaTensor,
}

/// Model-declared geometry of one grouped affine quantized projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineQuantizedConfig {
    /// Logical uncompressed input width.
    pub input_features: usize,
    /// Number of output rows.
    pub output_features: usize,
    /// Consecutive input values sharing scale and bias.
    pub group_size: usize,
    /// Packed checkpoint precision.
    pub bits: usize,
}

impl AffineQuantizedConfig {
    #[must_use]
    pub const fn new(
        input_features: usize,
        output_features: usize,
        group_size: usize,
        bits: usize,
    ) -> Self {
        Self {
            input_features,
            output_features,
            group_size,
            bits,
        }
    }

    pub(in crate::backend) fn spec(self) -> Result<AffineGemvSpec> {
        AffineGemvSpec::new(self.input_features, self.output_features, self.group_size, self.bits)
    }
}

/// Fixed-shape BF16-input affine grouped quantized GEMV.
#[derive(Clone, Debug)]
pub struct AffineQuantizedBf16Linear {
    operation: AffineQuantizedGemv,
    stream: Stream,
    matrices: usize,
}

impl AffineQuantizedBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        input_features: usize,
        output_features: usize,
        matrices: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<Self> {
        if matrices == 0 {
            return Err(Error::InvalidMatrixIndex { index: 0, matrices });
        }
        let spec = AffineGemvSpec::new(input_features, output_features, group_size, bits)?;
        Ok(Self {
            operation: AffineQuantizedGemv::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
            matrices,
        })
    }

    /// Enqueues one matrix or expert projection without allocation or host
    /// synchronization.
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
        let spec = self.operation.spec();
        let values_per_word = 32 / spec.bits;
        let packed = spec.input_features / values_per_word;
        let groups = spec.input_features / spec.group_size;
        validate_shape(
            tensors.weight,
            expected_shape(self.matrices, spec.output_features, packed),
        )?;
        validate_shape(
            tensors.scales,
            expected_shape(self.matrices, spec.output_features, groups),
        )?;
        validate_shape(
            tensors.biases,
            expected_shape(self.matrices, spec.output_features, groups),
        )?;
        let weight = tensors.weight.as_u32().ok_or_else(|| Error::DTypeMismatch {
            name: tensors.weight.name().into(),
            expected: "U32",
        })?;
        let scales = bf16_tensor(tensors.scales)?;
        let biases = bf16_tensor(tensors.biases)?;
        self.operation.execute(
            &self.stream,
            &mut AffineGemvLaunch {
                input,
                weight,
                scales,
                biases,
                output,
                matrix_index,
            },
        )
    }

    /// Number of BF16 elements required for one output vector.
    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.operation.spec().output_features
    }
}

pub(super) fn expected_shape(matrices: usize, output: usize, trailing: usize) -> Vec<usize> {
    if matrices == 1 {
        vec![output, trailing]
    } else {
        vec![matrices, output, trailing]
    }
}

pub(super) fn validate_shape(tensor: &CudaTensor, expected: Vec<usize>) -> Result<()> {
    if tensor.shape() != expected {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected,
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

pub(super) fn bf16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
