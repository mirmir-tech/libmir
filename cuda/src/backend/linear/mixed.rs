use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{CudaTensor, Error, Result};

/// Fixed-shape BF16 projection retaining FP32 accumulator output.
#[derive(Debug)]
pub struct Bf16Fp32Linear {
    plan: DenseMatmulPlan<bf16, f32>,
    stream: Stream,
    tokens: usize,
    input_features: usize,
    output_features: usize,
}

impl Bf16Fp32Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        Ok(Self {
            plan: DenseMatmulPlan::new(
                &backend.inner.context,
                &backend.inner.stream,
                DenseMatmulSpec::new(tokens, output_features, input_features)?,
            )?,
            stream: backend.inner.stream.clone(),
            tokens,
            input_features,
            output_features,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let expected_shape = [self.output_features, self.input_features];
        if weight.shape() != expected_shape {
            return Err(Error::InvalidLinearWeight {
                name: weight.name().into(),
                expected: expected_shape,
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        if output.len() != self.tokens * self.output_features {
            return Err(Error::InvalidTensorSize {
                name: "BF16 to FP32 linear output".into(),
                expected: self.tokens * self.output_features,
                actual: output.len(),
            });
        }
        Ok(self.plan.execute(&self.stream, input, weight, output, 1.0, 0.0)?)
    }
}
