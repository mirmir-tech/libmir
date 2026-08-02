use mircuda::{DeviceBuffer, bf16};

use super::super::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    AffineQuantizedWeight, CudaBackend,
};
use crate::{Result, backend::tuning::AffineProjectionExecution};

#[derive(Debug)]
pub(super) struct Candidate {
    pub(super) execution: AffineProjectionExecution,
    operation: Operation,
}

#[derive(Debug)]
enum Operation {
    Qmm(AffineQuantizedBf16Qmm),
    Gemv(AffineQuantizedBf16Linear),
}

impl Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        config: AffineQuantizedConfig,
        execution: AffineProjectionExecution,
    ) -> Result<Self> {
        let operation = match execution {
            AffineProjectionExecution::Qmm => {
                Operation::Qmm(AffineQuantizedBf16Qmm::new(backend, tokens, config, 1)?)
            },
            AffineProjectionExecution::Gemv => Operation::Gemv(AffineQuantizedBf16Linear::new(
                backend,
                config.input_features,
                config.output_features,
                1,
                config.group_size,
                config.bits,
            )?),
        };
        Ok(Self { execution, operation })
    }

    pub(super) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weights: &AffineQuantizedWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match &self.operation {
            Operation::Qmm(operation) => operation.execute(input, weights.tensors(), output, 0),
            Operation::Gemv(operation) => operation.execute(input, weights.tensors(), output, 0),
        }
    }
}
