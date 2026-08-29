use mircuda::{DeviceBuffer, bf16};

use super::SharedRoutedExecution;
use crate::{
    CudaBackend, Result,
    backend::shared_moe::weights::{AffineSharedExpertMoeWeights, RoutedSharedMoeWeights},
};

impl SharedRoutedExecution {
    pub(in crate::backend::shared_moe) fn nvfp4_prequant_scale(&self) -> Option<DeviceBuffer<f32>> {
        match self {
            Self::NvFp4(execution) => execution.prequant_scale(),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::shared_moe) fn execute_nvfp4_prequantized_residual_shared(
        &mut self,
        backend: &CudaBackend,
        weights: &AffineSharedExpertMoeWeights,
        input: &DeviceBuffer<bf16>,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        residual: &DeviceBuffer<bf16>,
        shared: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let (Self::NvFp4(execution), RoutedSharedMoeWeights::NvFp4(weights)) =
            (self, &weights.routed)
        else {
            return Err(crate::Error::InvalidExecutionPlan(
                "shared routed plan cannot fuse prequantized output",
            ));
        };
        execution.prepare_routing(backend, input)?;
        execution
            .execute_prequantized_residual_shared(packed, scales, weights, residual, shared, output)
    }
}
