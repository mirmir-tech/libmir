use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file,
};

use super::{DecisionWindow, require};
use crate::{Error, Result};

cuda_export!(DenseAttentionKernel = "libmir_decision_dense_attention"(
    qkv: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, lengths: &DeviceBuffer<u32>,
    length: u32, heads: u32, head_dim: u32, radius: u32, scale: f32,
));

/// Queries of one block and keys of one shared-memory tile.
const TILE: usize = 32;
const THREADS: u32 = 256;
/// Head width the shared-memory tiles are laid out for.
const MAX_HEAD_DIM: usize = 64;

/// Tiled attention of every token of a padded batch.
#[derive(Clone, Debug)]
pub struct DecisionDenseAttention {
    kernel: TypedKernel<DenseAttentionKernel>,
}

impl DecisionDenseAttention {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/decision/dense_attention_f32.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()? })
    }

    /// Writes `[rows × length, heads × head_dim]` outputs of a fused
    /// `[rows, length, 3 × heads × head_dim]` projection.
    pub fn execute(
        &self,
        stream: &Stream,
        (qkv, lengths): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
        (length, heads, head_dim, window): (usize, usize, usize, DecisionWindow),
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if head_dim == 0 || head_dim > MAX_HEAD_DIM || heads == 0 {
            return Err(Error::InvalidDecoderKernel("dense attention head geometry"));
        }
        let rows = lengths.len();
        require(qkv, rows * length * 3 * heads * head_dim, "dense attention qkv")?;
        require(output, rows * length * heads * head_dim, "dense attention output")?;
        let radius = match window {
            DecisionWindow::Full => u32::MAX,
            DecisionWindow::Band { radius } => u32::try_from(radius)?,
        };
        let scale = 1.0 / f32::from(u16::try_from(head_dim)?).sqrt();
        let launch = LaunchConfig {
            grid: (
                u32::try_from(length.div_ceil(TILE))?,
                u32::try_from(heads)?,
                u32::try_from(rows)?,
            ),
            block: (THREADS, 1, 1),
            shared_memory_bytes: 0,
        };
        let geometry = (u32::try_from(length)?, u32::try_from(heads)?, u32::try_from(head_dim)?);
        Ok(self.kernel.launch(
            stream,
            launch,
            (qkv, output, lengths, geometry.0, geometry.1, geometry.2, radius, scale),
        )?)
    }
}
