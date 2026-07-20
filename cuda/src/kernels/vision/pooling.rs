use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{CudaTensor, Error, Result};

cuda_export!(
    PoolKernel = "libmir_cuda_vision_pool_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        grid_height: u32, grid_width: u32, hidden: u32, kernel: u32,
    )
);
cuda_export!(
    StandardizeF32Kernel = "libmir_cuda_vision_standardize_f32"(
        input: &DeviceBuffer<bf16>, bias: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>,
        tokens: u32, hidden: u32,
    )
);
cuda_export!(
    StandardizeKernel = "libmir_cuda_vision_standardize_bf16"(
        input: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>,
        scale: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        tokens: u32, hidden: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionPoolSpec {
    pub grid_height: usize,
    pub grid_width: usize,
    pub hidden: usize,
    pub kernel: usize,
}

#[derive(Clone, Debug)]
pub struct VisionPool {
    pool: TypedKernel<PoolKernel>,
    standardize: TypedKernel<StandardizeKernel>,
    standardize_f32: TypedKernel<StandardizeF32Kernel>,
    spec: VisionPoolSpec,
}

impl VisionPool {
    pub fn compile(compiler: &Compiler, spec: VisionPoolSpec) -> Result<Self> {
        if spec.grid_height == 0
            || spec.grid_width == 0
            || spec.hidden == 0
            || spec.kernel == 0
            || !spec.grid_height.is_multiple_of(spec.kernel)
            || !spec.grid_width.is_multiple_of(spec.kernel)
        {
            return Err(Error::InvalidVisionKernel("invalid pooling geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/vision_pooling_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            pool: module.kernel()?,
            standardize: module.kernel()?,
            standardize_f32: module.kernel()?,
            spec,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("vision pool input", self.input_elements()?, input.len())?;
        require("vision pool output", self.output_elements()?, output.len())?;
        Ok(self.pool.launch(
            stream,
            launch(self.output_elements()?)?,
            (
                input,
                output,
                narrow(self.spec.grid_height)?,
                narrow(self.spec.grid_width)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.kernel)?,
            ),
        )?)
    }

    pub fn standardize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        scale: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = self.output_elements()?;
        require("vision standardize input", elements, input.len())?;
        require("vision standardize bias", self.spec.hidden, bias.len())?;
        require("vision standardize scale", self.spec.hidden, scale.len())?;
        require("vision standardize output", elements, output.len())?;
        Ok(self.standardize.launch(
            stream,
            launch(elements)?,
            (
                input,
                bias,
                scale,
                output,
                narrow(self.output_tokens())?,
                narrow(self.spec.hidden)?,
            ),
        )?)
    }

    pub fn standardize_tensors(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        bias: &CudaTensor,
        scale: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (bias.as_bf16(), scale.as_bf16()) {
            (Some(bias), Some(scale)) => self.standardize(stream, input, bias, scale, output),
            _ => match (bias.as_f32(), scale.as_f32()) {
                (Some(bias), Some(scale)) => {
                    let elements = self.output_elements()?;
                    require("vision standardize input", elements, input.len())?;
                    require("vision standardize bias", self.spec.hidden, bias.len())?;
                    require("vision standardize scale", self.spec.hidden, scale.len())?;
                    require("vision standardize output", elements, output.len())?;
                    Ok(self.standardize_f32.launch(
                        stream,
                        launch(elements)?,
                        (
                            input,
                            bias,
                            scale,
                            output,
                            narrow(self.output_tokens())?,
                            narrow(self.spec.hidden)?,
                        ),
                    )?)
                },
                _ => Err(Error::DTypeMismatch {
                    name: bias.name().into(),
                    expected: "matching BF16 or F32 standardization tensors",
                }),
            },
        }
    }

    pub fn output_elements(&self) -> Result<usize> {
        product(self.output_tokens(), self.spec.hidden)
    }

    fn input_elements(&self) -> Result<usize> {
        product(product(self.spec.grid_height, self.spec.grid_width)?, self.spec.hidden)
    }

    const fn output_tokens(&self) -> usize {
        (self.spec.grid_height / self.spec.kernel) * (self.spec.grid_width / self.spec.kernel)
    }
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    let threads = 256_usize;
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(threads))?, 1, 1),
        block: (narrow(threads)?, 1, 1),
        shared_memory_bytes: 0,
    })
}
