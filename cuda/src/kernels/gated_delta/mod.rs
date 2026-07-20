use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

mod convolution;
pub use convolution::{GatedDeltaConvolution, GatedDeltaConvolutionSpec};
mod transform;
pub use transform::{GatedDeltaTransformSpec, GatedDeltaTransforms};

cuda_export!(
    RecurrenceKernel = "libmir_cuda_gated_delta_recurrence_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
        alpha: &DeviceBuffer<bf16>, beta: &DeviceBuffer<bf16>, a_log: &DeviceBuffer<bf16>,
        dt_bias: &DeviceBuffer<bf16>, state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, key_heads: u32, value_heads: u32,
        key_dim: u32, value_dim: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaSpec {
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub struct GatedDeltaLaunch<'a> {
    pub query: &'a DeviceBuffer<bf16>,
    pub key: &'a DeviceBuffer<bf16>,
    pub value: &'a DeviceBuffer<bf16>,
    pub alpha: &'a DeviceBuffer<bf16>,
    pub beta: &'a DeviceBuffer<bf16>,
    pub a_log: &'a DeviceBuffer<bf16>,
    pub dt_bias: &'a DeviceBuffer<bf16>,
    pub state: &'a mut DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaRecurrence {
    kernel: TypedKernel<RecurrenceKernel>,
    spec: GatedDeltaSpec,
}

impl GatedDeltaRecurrence {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/gated_delta_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut GatedDeltaLaunch<'_>) -> Result<()> {
        let key = product(product(self.spec.tokens, self.spec.key_heads)?, self.spec.key_dim)?;
        let value =
            product(product(self.spec.tokens, self.spec.value_heads)?, self.spec.value_dim)?;
        let gates = product(self.spec.tokens, self.spec.value_heads)?;
        let state = self.state_elements()?;
        require("Gated Delta query", key, launch.query.len())?;
        require("Gated Delta key", key, launch.key.len())?;
        require("Gated Delta value", value, launch.value.len())?;
        require("Gated Delta alpha", gates, launch.alpha.len())?;
        require("Gated Delta beta", gates, launch.beta.len())?;
        require("Gated Delta A log", self.spec.value_heads, launch.a_log.len())?;
        require("Gated Delta time bias", self.spec.value_heads, launch.dt_bias.len())?;
        require("Gated Delta state", state, launch.state.len())?;
        require("Gated Delta output", value, launch.output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (1, narrow(self.spec.value_dim.div_ceil(4))?, narrow(self.spec.value_heads)?),
                block: (32, 4, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.query,
                launch.key,
                launch.value,
                launch.alpha,
                launch.beta,
                launch.a_log,
                launch.dt_bias,
                &mut *launch.state,
                &mut *launch.output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.key_heads)?,
                narrow(self.spec.value_heads)?,
                narrow(self.spec.key_dim)?,
                narrow(self.spec.value_dim)?,
            ),
        )?)
    }

    pub fn state_elements(&self) -> Result<usize> {
        product(product(self.spec.value_heads, self.spec.value_dim)?, self.spec.key_dim)
    }
}

fn validate(spec: GatedDeltaSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.key_heads == 0
        || spec.value_heads == 0
        || !spec.value_heads.is_multiple_of(spec.key_heads)
        || spec.key_dim == 0
        || !spec.key_dim.is_multiple_of(32)
        || spec.value_dim == 0
    {
        return Err(Error::InvalidDecoderKernel("invalid Gated Delta recurrence geometry"));
    }
    Ok(())
}
