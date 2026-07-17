use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DeviceBuffer, Stream, bf16};

use super::{
    CudaBackend, NvFp4Config, NvFp4Tensors, f32_tensor, output_elements, u8_tensor,
    validate_tensors,
};
use crate::{
    Result,
    kernels::{NvFp4Dequant, NvFp4DequantLaunch, NvFp4Spec},
};

#[derive(Debug)]
pub(super) struct ReferenceNvFp4 {
    plan: DenseMatmulPlan<bf16>,
    stream: Stream,
    pub(super) weight: DeviceBuffer<bf16>,
    tokens: usize,
    config: NvFp4Config,
}

impl ReferenceNvFp4 {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<Self> {
        let spec = NvFp4Spec::new(config.input_features, config.output_features)?;
        validate_tensors(spec, tensors)?;
        let mut weight =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, spec.elements()?)?;
        NvFp4Dequant::compile(&backend.inner.compiler, spec)?.execute(
            &backend.inner.stream,
            &mut NvFp4DequantLaunch {
                packed: u8_tensor(tensors.weight, "U8")?,
                block_scales: u8_tensor(tensors.weight_scale, "F8_E4M3")?,
                global_scale: f32_tensor(tensors.weight_scale_2)?,
                output: &mut weight,
            },
        )?;
        let matmul = DenseMatmulSpec::new(tokens, config.output_features, config.input_features)?;
        Ok(Self {
            plan: DenseMatmulPlan::new(&backend.inner.context, &backend.inner.stream, matmul)?,
            stream: backend.inner.stream.clone(),
            weight,
            tokens,
            config,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.plan.execute(&self.stream, input, &self.weight, output, 1.0, 0.0)?)
    }

    pub(super) fn output_elements(&self) -> Result<usize> {
        output_elements(self.tokens, self.config.output_features)
    }
}
