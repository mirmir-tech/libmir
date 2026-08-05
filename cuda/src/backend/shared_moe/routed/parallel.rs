use mircuda::{DeviceBuffer, bf16};

use super::SharedRoutedExecution;
use crate::{
    CudaBackend, Result,
    backend::shared_moe::{
        scratch::AffineSharedMoeScratch,
        weights::{AffineSharedExpertMoeWeights, RoutedSharedMoeWeights},
    },
};

impl SharedRoutedExecution {
    pub(in crate::backend::shared_moe) fn prepare_parallel(
        &mut self,
        backend: &CudaBackend,
        input: &DeviceBuffer<bf16>,
    ) -> Result<bool> {
        let Self::NvFp4(execution) = self else {
            return Ok(false);
        };
        execution.prepare_routing(backend, input)?;
        Ok(true)
    }

    pub(in crate::backend::shared_moe) fn execute_parallel_prepared(
        &mut self,
        weights: &AffineSharedExpertMoeWeights,
        input: &DeviceBuffer<bf16>,
        scratch: &mut AffineSharedMoeScratch,
    ) -> Result<()> {
        let (Self::NvFp4(execution), RoutedSharedMoeWeights::NvFp4(weights)) =
            (self, &weights.routed)
        else {
            return Err(crate::Error::InvalidExecutionPlan(
                "parallel routed execution was not prepared",
            ));
        };
        execution.execute_prepared(input, weights, &mut scratch.routed_output)
    }
}
