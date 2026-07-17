use mircuda::{DeviceBuffer, bf16};

use super::{Bf16Linear, Bf16VectorLinear, CudaBackend};
use crate::{CudaTensor, DenseExecution, DensePlanRequest, Result};

/// Planned BF16 projection retaining exactly one prepared implementation.
#[derive(Debug)]
pub enum Bf16Projection {
    Matrix(Bf16Linear),
    Vector(Bf16VectorLinear),
}

impl CudaBackend {
    /// Selects and prepares a BF16 projection through the central execution
    /// planner.
    pub fn prepare_bf16_projection(&self, request: DensePlanRequest) -> Result<Bf16Projection> {
        let plan = self.execution_planner().plan_dense(request)?;
        match plan.execution() {
            DenseExecution::Matrix => Ok(Bf16Projection::Matrix(Bf16Linear::new(
                self,
                request.tokens,
                request.input_features,
                request.output_features,
            )?)),
            DenseExecution::Vector => Ok(Bf16Projection::Vector(Bf16VectorLinear::new(
                self,
                request.input_features,
                request.output_features,
            )?)),
            DenseExecution::BlockFp8Vector => Err(crate::Error::InvalidExecutionPlan(
                "block FP8 plan requires prepared block FP8 weights",
            )),
            DenseExecution::Fp8Int4Vector => Err(crate::Error::InvalidExecutionPlan(
                "FP8 plus INT4 plan requires prepared quantized weights",
            )),
        }
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
        match self {
            Self::Matrix(operation) => operation.execute(input, weight, output),
            Self::Vector(operation) => operation.execute(input, weight, output),
        }
    }
}
