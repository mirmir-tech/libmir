#[cfg(test)]
mod tests;

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, KernelNode, LaunchConfig, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    RmsNormKernel = "libmir_cuda_rms_norm_bf16"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: u32,
        columns: u32,
        epsilon: f32,
    )
);

cuda_export!(
    RmsNormUnitKernel = "libmir_cuda_rms_norm_unit_bf16"(
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: u32,
        columns: u32,
        epsilon: f32,
    )
);

cuda_export!(
    pub RopeKernel = "libmir_cuda_rope_bf16"(
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        pairing_dim: u32,
        start_position: u32,
        theta: f32,
    )
);

#[derive(Clone, Debug)]
pub struct RmsNorm {
    kernel: TypedKernel<RmsNormKernel>,
    rows: usize,
    columns: usize,
    epsilon: f32,
}

#[derive(Clone, Debug)]
pub struct RmsNormUnit {
    kernel: TypedKernel<RmsNormUnitKernel>,
    rows: usize,
    columns: usize,
    epsilon: f32,
}

impl RmsNorm {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize, epsilon: f32) -> Result<Self> {
        if rows == 0 || columns == 0 || !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid RMSNorm geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/rms_norm_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            rows,
            columns,
            epsilon,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.rows, self.columns)?;
        require("RMSNorm input", elements, input.len())?;
        require("RMSNorm weight", self.columns, weight.len())?;
        require("RMSNorm output", elements, output.len())?;
        let config = LaunchConfig {
            grid: (narrow(self.rows)?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (input, weight, output, narrow(self.rows)?, narrow(self.columns)?, self.epsilon),
        )?)
    }
}

impl RmsNormUnit {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize, epsilon: f32) -> Result<Self> {
        if rows == 0 || columns == 0 || !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid unit RMSNorm geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/rms_norm_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            rows,
            columns,
            epsilon,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.rows, self.columns)?;
        require("unit RMSNorm input", elements, input.len())?;
        require("unit RMSNorm output", elements, output.len())?;
        let config = LaunchConfig {
            grid: (narrow(self.rows)?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (input, output, narrow(self.rows)?, narrow(self.columns)?, self.epsilon),
        )?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeSpec {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub pairing_dim: usize,
    pub theta: f32,
}

#[derive(Clone, Debug)]
pub struct Rope {
    kernel: TypedKernel<RopeKernel>,
    spec: RopeSpec,
}

type RopeArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
);

impl Rope {
    pub fn compile(compiler: &Compiler, spec: RopeSpec) -> Result<Self> {
        validate_rope(spec)?;
        let source = cuda_kernel_file!("../../../kernels/rope_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<()> {
        let (config, arguments) = self.launch(input, output, start_position)?;
        Ok(self.kernel.launch(stream, config, arguments)?)
    }

    pub fn execute_captured(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<KernelNode<RopeKernel>> {
        let (config, arguments) = self.launch(input, output, start_position)?;
        Ok(self.kernel.launch_captured(stream, config, arguments)?)
    }

    fn launch<'a>(
        &self,
        input: &'a DeviceBuffer<bf16>,
        output: &'a mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<(LaunchConfig, RopeArguments<'a>)> {
        let elements = product(product(self.spec.tokens, self.spec.heads)?, self.spec.head_dim)?;
        require("RoPE input", elements, input.len())?;
        require("RoPE output", elements, output.len())?;
        let threads = 256_u32;
        let config = LaunchConfig {
            grid: (narrow(elements.div_ceil(usize::try_from(threads)?))?, 1, 1),
            block: (threads, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok((
            config,
            (
                input,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.rotary_dim)?,
                narrow(self.spec.pairing_dim)?,
                narrow(start_position)?,
                self.spec.theta,
            ),
        ))
    }
}

fn validate_rope(spec: RopeSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.heads == 0
        || spec.head_dim == 0
        || spec.rotary_dim == 0
        || spec.rotary_dim > spec.head_dim
        || spec.pairing_dim < spec.rotary_dim
        || spec.pairing_dim > spec.head_dim
        || !spec.rotary_dim.is_multiple_of(2)
        || !spec.pairing_dim.is_multiple_of(2)
        || !spec.theta.is_finite()
        || spec.theta <= 0.0
    {
        Err(Error::InvalidDecoderKernel("invalid RoPE geometry"))
    } else {
        Ok(())
    }
}
