use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, KernelNode, LaunchConfig, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_file,
};

use super::geometry::narrow;
use crate::{Error, Result};

mod batch;
mod launch;
#[cfg(test)]
mod tests;

pub use batch::BatchedQkvPostprocess;

cuda_export!(
    pub QkvPostprocessKernel = "libmir_cuda_qkv_postprocess_bf16"(
        query_input: &DeviceBuffer<bf16>,
        key_input: &DeviceBuffer<bf16>,
        value_input: &DeviceBuffer<bf16>,
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        query_output: &mut DeviceBuffer<bf16>,
        key_output: &mut DeviceBuffer<bf16>,
        value_output: &mut DeviceBuffer<bf16>,
        tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        value_head_dim: u32,
        rotary_dim: u32,
        pairing_dim: u32,
        start_position: u32,
        theta: f32,
        epsilon: f32,
        separate_inputs: u32,
        normalize_query: u32,
        normalize_key: u32,
        normalize_value: u32,
    )
);

pub type QkvPostprocessArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    f32,
    u32,
    u32,
    u32,
    u32,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkvNormalization {
    pub query: bool,
    pub key: bool,
    pub value: bool,
}

impl QkvNormalization {
    pub const ALL: Self = Self { query: true, key: true, value: true };
    pub const NONE: Self = Self { query: false, key: false, value: false };
    pub const QUERY_KEY: Self = Self { query: true, key: true, value: false };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QkvPostprocessSpec {
    pub(crate) tokens: usize,
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) value_head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) pairing_dim: usize,
    pub(crate) theta: f32,
    pub(crate) epsilon: f32,
    pub(crate) normalization: QkvNormalization,
}

#[derive(Clone, Debug)]
pub struct QkvPostprocess {
    kernel: TypedKernel<QkvPostprocessKernel>,
    spec: QkvPostprocessSpec,
}

impl QkvPostprocess {
    pub(crate) fn compile(compiler: &Compiler, spec: QkvPostprocessSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/qkv_postprocess_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &self,
        stream: &Stream,
        packed: &DeviceBuffer<bf16>,
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        query_output: &mut DeviceBuffer<bf16>,
        key_output: &mut DeviceBuffer<bf16>,
        value_output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<()> {
        let (config, arguments) = self.launch(
            [packed, packed, packed],
            false,
            query_weight,
            key_weight,
            query_output,
            key_output,
            value_output,
            start_position,
        )?;
        Ok(self.kernel.launch(stream, config, arguments)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_separate(
        &self,
        stream: &Stream,
        inputs: [&DeviceBuffer<bf16>; 3],
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        query_output: &mut DeviceBuffer<bf16>,
        key_output: &mut DeviceBuffer<bf16>,
        value_output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<()> {
        let (config, arguments) = self.launch(
            inputs, true, query_weight, key_weight, query_output, key_output, value_output,
            start_position,
        )?;
        Ok(self.kernel.launch(stream, config, arguments)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_captured(
        &self,
        stream: &Stream,
        packed: &DeviceBuffer<bf16>,
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        query_output: &mut DeviceBuffer<bf16>,
        key_output: &mut DeviceBuffer<bf16>,
        value_output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<KernelNode<QkvPostprocessKernel>> {
        let (config, arguments) = self.launch(
            [packed, packed, packed],
            false,
            query_weight,
            key_weight,
            query_output,
            key_output,
            value_output,
            start_position,
        )?;
        Ok(self.kernel.launch_captured(stream, config, arguments)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_captured_separate(
        &self,
        stream: &Stream,
        inputs: [&DeviceBuffer<bf16>; 3],
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        query_output: &mut DeviceBuffer<bf16>,
        key_output: &mut DeviceBuffer<bf16>,
        value_output: &mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<KernelNode<QkvPostprocessKernel>> {
        let (config, arguments) = self.launch(
            inputs, true, query_weight, key_weight, query_output, key_output, value_output,
            start_position,
        )?;
        Ok(self.kernel.launch_captured(stream, config, arguments)?)
    }

    pub(crate) fn kernel(&self) -> TypedKernel<QkvPostprocessKernel> {
        self.kernel.clone()
    }

    pub(crate) fn config(&self) -> Result<LaunchConfig> {
        let heads = self
            .spec
            .query_heads
            .checked_add(self.spec.kv_heads.saturating_mul(2))
            .and_then(|heads| heads.checked_mul(self.spec.tokens))
            .ok_or(Error::InvalidDecoderKernel("QKV postprocess launch overflow"))?;
        Ok(LaunchConfig {
            grid: (narrow(heads)?, 1, 1),
            block: (128, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}

fn validate(spec: QkvPostprocessSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.query_heads == 0
        || spec.kv_heads == 0
        || spec.head_dim == 0
        || spec.value_head_dim == 0
        || spec.rotary_dim == 0
        || spec.rotary_dim > spec.head_dim
        || spec.pairing_dim < spec.rotary_dim
        || spec.pairing_dim > spec.head_dim
        || !spec.rotary_dim.is_multiple_of(2)
        || !spec.pairing_dim.is_multiple_of(2)
        || !spec.theta.is_finite()
        || spec.theta <= 0.0
        || !spec.epsilon.is_finite()
        || spec.epsilon < 0.0
    {
        Err(Error::InvalidDecoderKernel("invalid QKV postprocess geometry"))
    } else {
        Ok(())
    }
}
