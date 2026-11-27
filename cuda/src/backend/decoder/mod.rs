use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, Error, Result,
    kernels::{NormFormula, RmsNorm, Rope, RopeSpec},
};

/// Fixed-shape BF16 RMS normalization using checkpoint scale weights.
#[derive(Clone, Debug)]
pub struct RmsNormBf16 {
    operation: RmsNorm,
    stream: Stream,
    rows: usize,
    features: usize,
    epsilon: f32,
    formula: NormFormula,
}

impl RmsNormBf16 {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        rows: usize,
        features: usize,
        epsilon: f32,
    ) -> Result<Self> {
        Self::with_formula(backend, rows, features, epsilon, NormFormula::Rms)
    }

    /// A weighted row norm by `formula`; RMS unless the decoder declares
    /// `LayerNorm`.
    pub(in crate::backend) fn with_formula(
        backend: &CudaBackend,
        rows: usize,
        features: usize,
        epsilon: f32,
        formula: NormFormula,
    ) -> Result<Self> {
        Ok(Self {
            operation: RmsNorm::compile_formula(
                &backend.inner.compiler,
                rows,
                features,
                epsilon,
                formula,
            )?,
            stream: backend.inner.stream.clone(),
            rows,
            features,
            epsilon,
            formula,
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

    /// The residual-stream norm of an attention layer's model for `rows` rows.
    pub(in crate::backend) fn for_stream(
        backend: &CudaBackend,
        rows: usize,
        config: &crate::DecodeAttentionConfig,
    ) -> Result<Self> {
        let (hidden, epsilon) = (config.hidden_size, config.rms_norm_epsilon);
        Self::with_formula(backend, rows, hidden, epsilon, config.stream_norm)
    }

    /// The same norm for `rows` rows.
    pub(in crate::backend) fn resized(&self, backend: &CudaBackend, rows: usize) -> Result<Self> {
        Self::with_formula(backend, rows, self.features, self.epsilon, self.formula)
    }

    /// The input a sub-block's projections read: normalized by `weight`, or
    /// the residual stream itself when the block has no input norm
    /// (post-norm blocks normalize their outputs instead).
    pub(in crate::backend) fn apply(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: Option<&CudaTensor>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if let Some(weight) = weight {
            return self.execute(input, weight, output);
        }
        let elements = self.rows * self.features;
        Ok(self.stream.copy_device_range(input, 0..elements, output, 0)?)
    }

    /// The weight a fused norm-and-projection kernel takes.
    pub(in crate::backend) fn fused_weight<'a>(
        &self,
        weight: Option<&'a CudaTensor>,
    ) -> Result<&'a DeviceBuffer<bf16>> {
        if self.formula != NormFormula::Rms {
            return Err(Error::InvalidExecutionPlan("fused input norms are RMS norms"));
        }
        self.weight(weight.ok_or(Error::InvalidExecutionPlan(
            "a projection without an input norm cannot fuse one",
        ))?)
    }

    pub(in crate::backend) fn weight<'a>(
        &self,
        weight: &'a CudaTensor,
    ) -> Result<&'a DeviceBuffer<bf16>> {
        let expected = self.formula.weight_len(self.features);
        if weight.shape() != [expected] {
            return Err(Error::InvalidQuantizedTensor {
                name: weight.name().into(),
                expected: vec![expected],
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
