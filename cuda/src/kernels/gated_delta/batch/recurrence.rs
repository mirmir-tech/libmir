use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::GatedDeltaInputs;
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(
    BatchRecurrenceKernel = "libmir_cuda_gated_delta_batch_recurrence_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
        alpha: &DeviceBuffer<bf16>, beta: &DeviceBuffer<bf16>, a_log: &DeviceBuffer<bf16>,
        dt_bias: &DeviceBuffer<bf16>, state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, rows: u32, tokens: u32,
        key_heads: u32, value_heads: u32, key_dim: u32, value_dim: u32,
    )
);

#[derive(Clone, Debug)]
pub struct GatedDeltaBatchRecurrence {
    kernel: TypedKernel<BatchRecurrenceKernel>,
    spec: GatedDeltaBatchSpec,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaBatchSpec {
    pub rows: usize,
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl GatedDeltaBatchRecurrence {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaBatchSpec) -> Result<Self> {
        if spec.rows == 0
            || spec.tokens == 0
            || spec.key_heads == 0
            || spec.value_heads == 0
            || !spec.value_heads.is_multiple_of(spec.key_heads)
            || spec.key_dim == 0
            || !spec.key_dim.is_multiple_of(32)
            || spec.key_dim > 256
            || spec.value_dim == 0
        {
            return Err(Error::InvalidDecoderKernel("invalid batched Gated Delta recurrence"));
        }
        let source = cuda_kernel_file!("../../../../kernels/gated_delta_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        inputs: GatedDeltaInputs<'_>,
        state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let packed_tokens = product(self.spec.rows, self.spec.tokens)?;
        let key = product(product(packed_tokens, self.spec.key_heads)?, self.spec.key_dim)?;
        let value = product(product(packed_tokens, self.spec.value_heads)?, self.spec.value_dim)?;
        let gates = product(packed_tokens, self.spec.value_heads)?;
        let state_values = product(
            product(product(self.spec.rows, self.spec.value_heads)?, self.spec.value_dim)?,
            self.spec.key_dim,
        )?;
        require("batched Gated Delta query", key, inputs.query.len())?;
        require("batched Gated Delta key", key, inputs.key.len())?;
        require("batched Gated Delta value", value, inputs.value.len())?;
        require("batched Gated Delta alpha", gates, inputs.alpha.len())?;
        require("batched Gated Delta beta", gates, inputs.beta.len())?;
        require("batched Gated Delta A log", self.spec.value_heads, inputs.a_log.len())?;
        require("batched Gated Delta time bias", self.spec.value_heads, inputs.dt_bias.len())?;
        require("batched Gated Delta state", state_values, state.len())?;
        require("batched Gated Delta output", value, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.rows)?,
                    narrow(self.spec.value_dim.div_ceil(4))?,
                    narrow(self.spec.value_heads)?,
                ),
                block: (32, 4, 1),
                shared_memory_bytes: 0,
            },
            (
                inputs.query,
                inputs.key,
                inputs.value,
                inputs.alpha,
                inputs.beta,
                inputs.a_log,
                inputs.dt_bias,
                state,
                output,
                narrow(self.spec.rows)?,
                narrow(self.spec.tokens)?,
                narrow(self.spec.key_heads)?,
                narrow(self.spec.value_heads)?,
                narrow(self.spec.key_dim)?,
                narrow(self.spec.value_dim)?,
            ),
        )?)
    }
}
