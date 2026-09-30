use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file,
};

use super::require;
use crate::{Error, Result};

cuda_export!(AttentionKernel = "libmir_decision_attention"(
    qkv: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, queries: &DeviceBuffer<u32>,
    lengths: &DeviceBuffer<u32>, query_count: u32, length: u32, heads: u32, head_dim: u32,
    radius: u32, scale: f32,
));

/// Largest head width one warp holds in registers.
const MAX_HEAD_DIM: usize = 128;
/// Kernel encoding of a window without a band.
const FULL_WINDOW: u32 = u32::MAX;

/// Keys a query may attend to besides the sequence-length mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionWindow {
    Full,
    Band { radius: usize },
}

/// A fused `[batch, length, 3 × heads × head_dim]` projection and the flat
/// `batch × length` indices of the queries to evaluate.
pub struct DecisionAttentionInput<'a> {
    pub qkv: &'a DeviceBuffer<f32>,
    pub queries: &'a DeviceBuffer<u32>,
    pub lengths: &'a DeviceBuffer<u32>,
    pub length: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub window: DecisionWindow,
}

impl DecisionWindow {
    pub(super) fn kernel_radius(self) -> Result<u32> {
        Ok(match self {
            Self::Full => FULL_WINDOW,
            Self::Band { radius } => u32::try_from(radius)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DecisionAttention {
    kernel: TypedKernel<AttentionKernel>,
}

impl DecisionAttention {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/decision/attention_f32.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()? })
    }

    /// Writes `[queries, heads × head_dim]` attention outputs.
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DecisionAttentionInput<'_>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.head_dim == 0 || input.head_dim > MAX_HEAD_DIM || input.heads == 0 {
            return Err(Error::InvalidDecoderKernel("decision attention head geometry"));
        }
        let hidden = input.heads * input.head_dim;
        let count = input.queries.len();
        require(output, count * hidden, "decision attention output")?;
        if input.qkv.len() != input.lengths.len() * input.length * 3 * hidden {
            return Err(Error::InvalidDecoderKernel("decision attention qkv geometry"));
        }
        let radius = input.window.kernel_radius()?;
        let scale = 1.0 / f32::from(u16::try_from(input.head_dim)?).sqrt();
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(count)?, u32::try_from(input.heads)?, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input.qkv,
                output,
                input.queries,
                input.lengths,
                u32::try_from(count)?,
                u32::try_from(input.length)?,
                u32::try_from(input.heads)?,
                u32::try_from(input.head_dim)?,
                radius,
                scale,
            ),
        )?)
    }
}
