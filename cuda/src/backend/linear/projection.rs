use mircuda::{DeviceBuffer, bf16};

use super::{AutoBf16Plan, CudaBackend};
use crate::{CudaTensor, DensePlanRequest, Error, Result};

/// Planned BF16 projection retaining exactly one prepared implementation.
#[derive(Debug)]
pub struct Bf16Projection {
    operation: AutoBf16Plan,
}

impl CudaBackend {
    /// Selects and prepares a BF16 projection through the central execution
    /// planner.
    pub fn prepare_bf16_projection(&self, request: DensePlanRequest) -> Result<Bf16Projection> {
        Ok(Bf16Projection {
            operation: AutoBf16Plan::new(self, request)?,
        })
    }
}

impl Bf16Projection {
    /// Enqueues the selected projection without allocation or synchronization.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let expected =
            [self.operation.request().output_features, self.operation.request().input_features];
        if weight.shape() != expected {
            return Err(Error::InvalidLinearWeight {
                name: weight.name().into(),
                expected,
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        self.operation.execute(input, weight, output)
    }
}
