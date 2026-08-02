use mircuda::{CublasLtBf16Plan, CublasLtBf16Spec, DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{CudaTensor, Error, Result};

/// Fixed-shape vendor BF16 projection.
#[derive(Debug)]
pub struct Bf16VendorLinear {
    plan: CublasLtBf16Plan,
    stream: Stream,
    tokens: usize,
    input_features: usize,
    output_features: usize,
}

impl Bf16VendorLinear {
    pub fn new(
        backend: &CudaBackend,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        Ok(Self {
            plan: CublasLtBf16Plan::new(
                &backend.inner.context,
                &backend.inner.stream,
                CublasLtBf16Spec::new(tokens, output_features, input_features)?,
            )?,
            stream: backend.inner.stream.clone(),
            tokens,
            input_features,
            output_features,
        })
    }

    /// Enqueues one vendor projection without allocation or synchronization.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if input.len() != self.tokens * self.input_features
            || output.len() != self.tokens * self.output_features
        {
            return Err(Error::InvalidDecoderKernel("vendor BF16 activation differs from plan"));
        }
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
        Ok(self.plan.execute(&self.stream, input, weight, output, 1.0, 0.0)?)
    }
}
