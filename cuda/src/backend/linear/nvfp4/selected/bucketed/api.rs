use mircuda::DeviceBuffer;

use super::{BucketedNvFp4MoeBf16, CudaBackend, GatedActivation, NvFp4ExpertBank};
use crate::Result;

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn prepare_bucketed_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<BucketedNvFp4MoeBf16> {
        BucketedNvFp4MoeBf16::new(self, tokens, selected, activation, gate, up, down)
    }
}

impl BucketedNvFp4MoeBf16 {
    pub(in crate::backend) fn prequant_scale(&self) -> Option<DeviceBuffer<f32>> {
        self.gate_up.prequant_scale()
    }

    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.output_elements
    }
}
