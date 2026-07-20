use mircuda::{DeviceBuffer, bf16};

use crate::{
    AffineQuantizedBf16Qmm, AffineQuantizedConfig, AffineQuantizedWeight, CudaBackend, Result,
};

#[derive(Clone, Debug)]
pub(in crate::backend) struct AffineProjection {
    operation: AffineQuantizedBf16Qmm,
    weights: AffineQuantizedWeight,
}

impl AffineProjection {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        group_size: usize,
        bits: usize,
        weights: &AffineQuantizedWeight,
    ) -> Result<Self> {
        Ok(Self {
            operation: backend.prepare_affine_quantized_bf16_qmm(
                tokens,
                AffineQuantizedConfig::new(input, output, group_size, bits),
                1,
            )?,
            weights: weights.clone(),
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.operation.execute(input, self.weights.tensors(), output, 0)
    }
}
