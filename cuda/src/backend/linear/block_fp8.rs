use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, Error, Result,
    kernels::{
        BlockFp8LinearKernels, BlockFp8LinearSpec, Fp8OutputKernels, Fp8OutputSpec,
        Fp8ResidualWeightBuffers,
    },
};

/// Model-owned block-scaled E4M3 weight for BF16-input decode projection.
#[derive(Clone, Debug)]
pub struct BlockFp8LinearWeight {
    kernels: BlockFp8LinearKernels,
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<f32>,
    spec: BlockFp8LinearSpec,
}

/// Model-owned E4M3 base plus block-scaled INT4 correction.
#[derive(Clone, Debug)]
pub struct Fp8ResidualLinearWeight {
    kernels: Fp8OutputKernels,
    weight: DeviceBuffer<u8>,
    row_scales: DeviceBuffer<f32>,
    residual: DeviceBuffer<u8>,
    residual_scales: DeviceBuffer<f32>,
}

impl CudaBackend {
    pub(in crate::backend) fn prepare_block_fp8_linear_weight(
        &self,
        source: &CudaTensor,
    ) -> Result<BlockFp8LinearWeight> {
        let [output_features, input_features] = source.shape() else {
            return Err(Error::InvalidLinearWeight {
                name: source.name().into(),
                expected: [0, 0],
                actual: source.shape().to_vec(),
            });
        };
        let source_weight = source.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: source.name().into(),
            expected: "BF16",
        })?;
        let spec = BlockFp8LinearSpec::new(*input_features, *output_features)?;
        let kernels = BlockFp8LinearKernels::compile(&self.inner.compiler, spec)?;
        let mut weight =
            self.inner.pool.allocate::<u8>(&self.inner.stream, spec.weight_elements()?)?;
        let mut scales =
            self.inner.pool.allocate::<f32>(&self.inner.stream, spec.scale_elements()?)?;
        kernels.quantize(&self.inner.stream, source_weight, &mut weight, &mut scales)?;
        Ok(BlockFp8LinearWeight { kernels, weight, scales, spec })
    }

    pub(in crate::backend) fn prepare_fp8_residual_linear_weight(
        &self,
        source: &CudaTensor,
    ) -> Result<Fp8ResidualLinearWeight> {
        let [output_features, input_features] = source.shape() else {
            return Err(Error::InvalidLinearWeight {
                name: source.name().into(),
                expected: [0, 0],
                actual: source.shape().to_vec(),
            });
        };
        let source_weight = source.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: source.name().into(),
            expected: "BF16",
        })?;
        let spec = Fp8OutputSpec::new(*input_features, *output_features)?;
        let kernels = Fp8OutputKernels::compile(&self.inner.compiler, spec)?;
        let mut weight =
            self.inner.pool.allocate::<u8>(&self.inner.stream, spec.weight_elements()?)?;
        let mut block_scales = self
            .inner
            .pool
            .allocate::<f32>(&self.inner.stream, spec.weight_scale_elements()?)?;
        let mut row_scales =
            self.inner.pool.allocate::<f32>(&self.inner.stream, spec.output_features)?;
        let mut residual =
            self.inner.pool.allocate::<u8>(&self.inner.stream, spec.residual_elements()?)?;
        let mut residual_scales = self
            .inner
            .pool
            .allocate::<f32>(&self.inner.stream, spec.residual_scale_elements()?)?;
        let mut buffers = Fp8ResidualWeightBuffers::new(
            &mut weight,
            &mut block_scales,
            &mut row_scales,
            &mut residual,
            &mut residual_scales,
        );
        kernels.quantize_weight_residual(&self.inner.stream, source_weight, &mut buffers)?;
        Ok(Fp8ResidualLinearWeight {
            kernels,
            weight,
            row_scales,
            residual,
            residual_scales,
        })
    }
}

impl BlockFp8LinearWeight {
    pub(in crate::backend) fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.kernels.project(stream, input, &self.weight, &self.scales, output)
    }

    #[must_use]
    pub const fn spec(&self) -> BlockFp8LinearSpec {
        self.spec
    }
}

impl Fp8ResidualLinearWeight {
    pub(in crate::backend) fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.kernels.project_residual(
            stream,
            input,
            &self.weight,
            &self.row_scales,
            &self.residual,
            &self.residual_scales,
            output,
        )
    }
}
