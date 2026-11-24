//! Absorbed multi-head latent attention over a paged latent cache.
//!
//! A cache row holds a token's normalized latent and its rotated key. Queries
//! are absorbed into the latent space (`q_nope · W_UK`) so attention runs as
//! multi-query attention over those rows, and the latent output is expanded
//! per head (`· W_UVᵀ`). The same kernels serve decode, chunked prefill and
//! batches: every query token carries its sequence and its position and
//! attends to the tokens up to it; its own row goes to its slot.

mod absorb;
mod execute;
mod expanded;
mod kernels;
#[cfg(test)]
mod tests;

pub use absorb::{LatentAbsorption, LatentUpLayout, LatentUpWeights};
pub use execute::{LatentAttentionInputs, LatentAttentionLayer, LatentCacheView};
pub use expanded::{
    LatentExpandedMetadata, LatentExpandedRow, LatentExpandedRows, LatentExpansion,
    LatentExpansionBuffers,
};
use kernels::{
    ExpandedKernels, HEADS_PER_BLOCK, LatentMergeKernel, LatentMmaKernel, LatentMmaPairKernel,
    LatentPrepareKernel, LatentSplit, MMA_GROUPS, MMA_HEADS, MMA_SHARED_LIMIT, MMA_THREADS,
    mma_shared,
};
use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, MemoryPool, Stream, TypedKernel, bf16, cuda_kernel_file,
};
use models::layout::RotaryPairing;

use super::geometry::product;
use crate::{Error, Result};

/// Fixed geometry of one latent attention layer and its execution limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentAttentionSpec {
    pub heads: usize,
    /// Width of the cached latent, also the absorbed query/value width.
    pub latent_rank: usize,
    pub rope_dim: usize,
    pub nope_dim: usize,
    pub value_dim: usize,
    pub rotary_pairing: RotaryPairing,
    pub block_size: usize,
    /// Query tokens one launch may carry.
    pub max_queries: usize,
    /// Partitions of every query's visible tokens, merged afterwards.
    pub splits: usize,
}

impl LatentAttentionSpec {
    /// Elements of one cache or attention-query row.
    #[must_use]
    pub const fn row_width(&self) -> usize {
        self.latent_rank + self.rope_dim
    }

    /// Bytes of a BF16 page pool of `blocks` blocks.
    pub fn page_bytes(&self, blocks: usize) -> Result<usize> {
        product(product(product(blocks, self.block_size)?, self.row_width())?, 2)
    }

    fn validate(&self) -> Result<()> {
        let valid = self.heads > 0
            && self.latent_rank > 0
            && self.latent_rank.is_multiple_of(64)
            && self.rope_dim > 0
            && self.rope_dim.is_multiple_of(8)
            && self.nope_dim > 0
            && self.value_dim > 0
            && self.block_size > 0
            && self.max_queries > 0
            && self.splits > 0
            // The merge kernel weighs at most 64 splits.
            && self.splits <= 64
            // The kernel stages eight query rows and a 32-row tile in 48 KiB
            // of static shared memory.
            && (HEADS_PER_BLOCK + 32) * (self.row_width() + 2) * 2 <= 48 * 1024;
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidPagedKv("invalid latent attention geometry"))
        }
    }

    /// Whether the tensor-core kernels serve this geometry: whole groups of
    /// sixteen heads, a row split into four 16-column quarters (as the kernel
    /// source's `#if` requires), and two key tiles with every group's score
    /// exchange within one block's opt-in shared memory.
    const fn tensor_cores(&self) -> bool {
        let row = self.row_width();
        self.heads.is_multiple_of(MMA_HEADS)
            && row.is_multiple_of(64)
            && mma_shared(row, MMA_GROUPS) <= MMA_SHARED_LIMIT
    }
}

/// Compiled latent attention kernels for one [`LatentAttentionSpec`].
#[derive(Clone, Debug)]
pub struct LatentAttention {
    prepare: TypedKernel<LatentPrepareKernel>,
    split: LatentSplit,
    merge: TypedKernel<LatentMergeKernel>,
    expanded: ExpandedKernels,
    spec: LatentAttentionSpec,
}

impl LatentAttention {
    pub fn compile(compiler: &Compiler, spec: LatentAttentionSpec) -> Result<Self> {
        spec.validate()?;
        let mut extra_options = vec![
            format!("-DLIBMIR_LATENT_MMA_GROUPS={MMA_GROUPS}"),
            format!("-DLIBMIR_LATENT_RANK={}", spec.latent_rank),
            format!("-DLIBMIR_LATENT_ROPE={}", spec.rope_dim),
        ];
        if spec.rotary_pairing == RotaryPairing::Interleaved {
            extra_options.push("-DLIBMIR_LATENT_INTERLEAVED=1".into());
        }
        let options = CompileOptions {
            extra_options,
            ..CompileOptions::default()
        };
        let source = cuda_kernel_file!("../../../kernels/latent_attention_bf16.cu");
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            prepare: module.kernel()?,
            split: if !spec.tensor_cores() {
                LatentSplit::Warp(module.kernel()?)
            } else if spec.splits == 1 {
                // Prefill: pairs of row groups share every key tile.
                let kernel: TypedKernel<LatentMmaPairKernel> = module.kernel()?;
                kernel.set_max_dynamic_shared_memory_bytes(u32::try_from(mma_shared(
                    spec.row_width(),
                    MMA_GROUPS,
                ))?)?;
                LatentSplit::TensorCorePairs(kernel)
            } else {
                let kernel: TypedKernel<LatentMmaKernel> = module.kernel()?;
                kernel.set_max_dynamic_shared_memory_bytes(u32::try_from(mma_shared(
                    spec.row_width(),
                    1,
                ))?)?;
                LatentSplit::TensorCore(kernel)
            },
            merge: module.kernel()?,
            expanded: ExpandedKernels::load(&module)?,
            spec,
        })
    }

    #[must_use]
    pub const fn spec(&self) -> LatentAttentionSpec {
        self.spec
    }

    /// Allocates the per-launch buffers for `max_queries` query tokens.
    pub fn workspace(
        &self,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<LatentAttentionWorkspace> {
        let spec = self.spec;
        let rows = product(spec.max_queries, spec.heads)?;
        let partials = product(rows, spec.splits)?;
        Ok(LatentAttentionWorkspace {
            query: pool.allocate(stream, product(rows, spec.row_width())?)?,
            latent_output: pool.allocate(stream, product(rows, spec.latent_rank)?)?,
            values: pool.allocate(stream, product(partials, spec.latent_rank)?)?,
            statistics: pool.allocate(stream, product(partials, 2)?)?,
        })
    }
}

/// Scratch the latent attention pipeline writes between its stages.
#[derive(Debug)]
pub struct LatentAttentionWorkspace {
    /// `[queries, heads, latent_rank + rope_dim]` absorbed and rotated queries.
    pub(crate) query: DeviceBuffer<bf16>,
    /// `[queries, heads, latent_rank]` merged latent attention output.
    pub(crate) latent_output: DeviceBuffer<bf16>,
    pub(crate) values: DeviceBuffer<f32>,
    pub(crate) statistics: DeviceBuffer<f32>,
}
