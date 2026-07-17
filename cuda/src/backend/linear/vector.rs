use mircuda::{DenseVectorPlan, DenseVectorSpec, DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{CudaTensor, Error, Result};

/// Fixed-shape BF16 `weight × input` plan for one decode row.
#[derive(Debug)]
pub struct Bf16VectorLinear {
    plan: DenseVectorPlan<bf16>,
    stream: Stream,
    input_features: usize,
    output_features: usize,
}

impl Bf16VectorLinear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        Ok(Self {
            plan: DenseVectorPlan::new(
                &backend.inner.context,
                &backend.inner.stream,
                DenseVectorSpec::new(output_features, input_features)?,
            )?,
            stream: backend.inner.stream.clone(),
            input_features,
            output_features,
        })
    }

    /// Enqueues one vector projection without allocation or synchronization.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
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
        Ok(self.plan.execute(&self.stream, input, weight, output, 1.0, 0.0)?)
    }
}
