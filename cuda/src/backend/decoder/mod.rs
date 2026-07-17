use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, Error, Result,
    kernels::{RmsNorm, Rope, RopeSpec},
};

/// Fixed-shape BF16 RMS normalization using checkpoint scale weights.
#[derive(Clone, Debug)]
pub struct RmsNormBf16 {
    operation: RmsNorm,
    stream: Stream,
    features: usize,
    epsilon: f32,
}

impl RmsNormBf16 {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        rows: usize,
        features: usize,
        epsilon: f32,
    ) -> Result<Self> {
        Ok(Self {
            operation: RmsNorm::compile(&backend.inner.compiler, rows, features, epsilon)?,
            stream: backend.inner.stream.clone(),
            features,
            epsilon,
        })
    }

    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let weight = self.weight(weight)?;
        self.operation.execute(&self.stream, input, weight, output)
    }

    pub(in crate::backend) fn weight<'a>(
        &self,
        weight: &'a CudaTensor,
    ) -> Result<&'a DeviceBuffer<bf16>> {
        if weight.shape() != [self.features] {
            return Err(Error::InvalidQuantizedTensor {
                name: weight.name().into(),
                expected: vec![self.features],
                actual: weight.shape().to_vec(),
            });
        }
        weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })
    }

    #[must_use]
    pub(in crate::backend) const fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

/// Standard half-split `RoPE` with optional partial rotary dimensions.
#[derive(Clone, Debug)]
pub struct RopeBf16 {
    operation: Rope,
    stream: Stream,
}

impl RopeBf16 {
    pub(in crate::backend) fn new(backend: &CudaBackend, spec: RopeSpec) -> Result<Self> {
        Ok(Self {
            operation: Rope::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
        })
    }

    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<()> {
        self.operation.execute(&self.stream, input, output, start_position)
    }
}
