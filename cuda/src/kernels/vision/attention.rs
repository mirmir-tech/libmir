use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    AttentionKernel = "libmir_cuda_vision_attention_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        tokens: u32, query_heads: u32, kv_heads: u32, head_dim: u32, scale: f32,
    )
);
cuda_export!(
    SpatialRopeKernel = "libmir_cuda_vision_spatial_rope_bf16"(
        input: &DeviceBuffer<bf16>, positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, heads: u32,
        head_dim: u32, theta: f32,
    )
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionAttentionSpec {
    pub tokens: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct VisionAttention {
    kernel: TypedKernel<AttentionKernel>,
    spec: VisionAttentionSpec,
}

#[derive(Clone, Debug)]
pub struct VisionSpatialRope {
    kernel: TypedKernel<SpatialRopeKernel>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
}

impl VisionAttention {
    pub fn compile(compiler: &Compiler, spec: VisionAttentionSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/vision_attention_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key: &DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let query_elements =
            product(product(self.spec.tokens, self.spec.query_heads)?, self.spec.head_dim)?;
        let kv_elements =
            product(product(self.spec.tokens, self.spec.kv_heads)?, self.spec.head_dim)?;
        require("vision attention query", query_elements, query.len())?;
        require("vision attention key", kv_elements, key.len())?;
        require("vision attention value", kv_elements, value.len())?;
        require("vision attention output", query_elements, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.tokens)?, narrow(self.spec.query_heads)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key,
                value,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                self.spec.scale,
            ),
        )?)
    }
}

impl VisionSpatialRope {
    pub fn compile(
        compiler: &Compiler,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<Self> {
        if tokens == 0
            || heads == 0
            || !head_dim.is_multiple_of(4)
            || head_dim > 256
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(Error::InvalidVisionKernel("invalid spatial RoPE geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/vision_attention_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            tokens,
            heads,
            head_dim,
            theta,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(product(self.tokens, self.heads)?, self.head_dim)?;
        require("vision RoPE input", elements, input.len())?;
        require("vision RoPE positions", self.tokens * 2, positions.len())?;
        require("vision RoPE output", elements, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                positions,
                output,
                narrow(self.tokens)?,
                narrow(self.heads)?,
                narrow(self.head_dim)?,
                self.theta,
            ),
        )?)
    }
}

fn validate(spec: VisionAttentionSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.query_heads == 0
        || spec.kv_heads == 0
        || !spec.query_heads.is_multiple_of(spec.kv_heads)
        || spec.head_dim == 0
        || spec.head_dim > 256
        || !spec.scale.is_finite()
        || spec.scale <= 0.0
    {
        Err(Error::InvalidVisionKernel("invalid attention geometry"))
    } else {
        Ok(())
    }
}
