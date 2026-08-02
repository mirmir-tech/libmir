use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result};

mod experts;
mod qkv;
mod routes;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use experts::{DownKernel, DownMlxKernel, GateUpKernel, GateUpMlxKernel};
use qkv::{QkvKernel, QkvSplitKernel, RopeKernel};
use routes::{DownMlxRoutesKernel, DownRoutesKernel, ReduceRoutesKernel};

cuda_export!(BiasKernel = "libmir_cuda_clamped_routed_add_bias_bf16"(
    input: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
));

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClampedRoutedSpec {
    pub tokens: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub theta: f32,
    pub factor: f32,
    pub initial_context: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub swiglu_limit: f32,
}

#[derive(Clone, Debug)]
pub struct ClampedRoutedKernels {
    qkv: TypedKernel<QkvKernel>,
    qkv_split: TypedKernel<QkvSplitKernel>,
    rope: TypedKernel<RopeKernel>,
    bias: TypedKernel<BiasKernel>,
    gate_up: TypedKernel<GateUpKernel>,
    down: TypedKernel<DownKernel>,
    down_routes: TypedKernel<DownRoutesKernel>,
    gate_up_mlx: TypedKernel<GateUpMlxKernel>,
    down_mlx: TypedKernel<DownMlxKernel>,
    down_mlx_routes: TypedKernel<DownMlxRoutesKernel>,
    reduce_routes: TypedKernel<ReduceRoutesKernel>,
    spec: ClampedRoutedSpec,
}

impl ClampedRoutedKernels {
    pub(crate) fn compile(compiler: &Compiler, spec: ClampedRoutedSpec) -> Result<Self> {
        if spec.tokens == 0
            || spec.hidden == 0
            || spec.intermediate == 0
            || !spec.hidden.is_multiple_of(32)
            || !spec.intermediate.is_multiple_of(32)
            || spec.head_dim == 0
            || !spec.head_dim.is_multiple_of(2)
            || spec.top_k == 0
        {
            return Err(Error::InvalidDecoderKernel("invalid clamped-routed CUDA geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../../../kernels/clamped_routed_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            qkv: module.kernel()?,
            qkv_split: module.kernel()?,
            rope: module.kernel()?,
            bias: module.kernel()?,
            gate_up: module.kernel()?,
            down: module.kernel()?,
            down_routes: module.kernel()?,
            gate_up_mlx: module.kernel()?,
            down_mlx: module.kernel()?,
            down_mlx_routes: module.kernel()?,
            reduce_routes: module.kernel()?,
            spec,
        })
    }

    pub(crate) fn add_bias(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        columns: usize,
    ) -> Result<()> {
        let total = self.spec.tokens * columns;
        Ok(self.bias.launch(
            stream,
            linear_launch(total)?,
            (input, bias, output, narrow(self.spec.tokens)?, narrow(columns)?),
        )?)
    }
}

pub(super) fn linear_launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}

pub(super) fn narrow(value: usize) -> Result<u32> {
    Ok(u32::try_from(value)?)
}
