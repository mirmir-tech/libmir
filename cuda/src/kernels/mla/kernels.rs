//! The kernel exports of `kernels/latent_attention_bf16.cu` and the
//! tensor-core launch geometry.

use mircuda::{DeviceBuffer, Module, TypedKernel, bf16, cuda_export};

use crate::Result;

cuda_export!(
    pub(crate) LatentPrepareKernel = "libmir_cuda_latent_prepare_bf16"(
        query: &DeviceBuffer<bf16>, latent: &DeviceBuffer<bf16>,
        norm_weight: &DeviceBuffer<bf16>, positions: &DeviceBuffer<u32>,
        inverse_frequencies: &DeviceBuffer<f32>, slot_mapping: &DeviceBuffer<u32>,
        attention_query: &mut DeviceBuffer<bf16>, pages: &mut DeviceBuffer<u8>,
        tokens: u32, heads: u32, nope_dim: u32, epsilon: f32, rope_scale: f32,
    )
);

cuda_export!(
    pub(crate) LatentSplitKernel = "libmir_cuda_latent_attention_split_bf16"(
        query: &DeviceBuffer<bf16>, pages: &DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>, query_sequences: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>, partial_values: &mut DeviceBuffer<f32>,
        partial_statistics: &mut DeviceBuffer<f32>, queries: u32, heads: u32,
        table_stride: u32, block_size: u32, scale: f32, temperature_beta: f32,
        temperature_window: u32,
    )
);

cuda_export!(
    pub(crate) LatentMmaKernel = "libmir_cuda_latent_attention_mma_bf16"(
        query: &DeviceBuffer<bf16>, pages: &DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>, query_sequences: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>, partial_values: &mut DeviceBuffer<f32>,
        partial_statistics: &mut DeviceBuffer<f32>, queries: u32, heads: u32,
        table_stride: u32, block_size: u32, scale: f32, temperature_beta: f32,
        temperature_window: u32,
    )
);

cuda_export!(
    pub(crate) LatentMmaPairKernel = "libmir_cuda_latent_attention_mma_pair_bf16"(
        query: &DeviceBuffer<bf16>, pages: &DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>, query_sequences: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>, partial_values: &mut DeviceBuffer<f32>,
        partial_statistics: &mut DeviceBuffer<f32>, queries: u32, heads: u32,
        table_stride: u32, block_size: u32, scale: f32, temperature_beta: f32,
        temperature_window: u32,
    )
);

cuda_export!(
    pub(crate) LatentMergeKernel = "libmir_cuda_latent_attention_merge_bf16"(
        partial_values: &DeviceBuffer<f32>, partial_statistics: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, rows: u32, splits: u32,
    )
);

/// Heads one warp-kernel block serves; it must match `HEADS_PER_BLOCK` in the
/// kernel source.
pub(super) const HEADS_PER_BLOCK: usize = 8;
/// Heads of one tensor-core row group (`MMA_ROWS`) and its threads.
pub(super) const MMA_HEADS: usize = 16;
pub(super) const MMA_THREADS: usize = 128;
/// Row groups of one prefill block (`LIBMIR_LATENT_MMA_GROUPS`).
pub(super) const MMA_GROUPS: usize = 2;
/// Opt-in dynamic shared memory of one block on compute capability 12.x.
pub(super) const MMA_SHARED_LIMIT: usize = 99 * 1024;

/// Dynamic shared memory of a tensor-core block of `groups` row groups: the
/// 32-row key tile and each group's score exchange. Double-buffering the tile
/// measured slower: it halves the blocks resident on an SM.
pub(super) const fn mma_shared(row_width: usize, groups: usize) -> usize {
    32 * (row_width + 8) * 2 + groups * 4 * MMA_HEADS * 40 * 4
}

cuda_export!(
    pub(crate) LatentGatherKernel = "libmir_cuda_latent_gather_bf16"(
        pages: &DeviceBuffer<u8>, block_tables: &DeviceBuffer<u32>,
        contexts: &DeviceBuffer<u32>, regions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>, table_stride: u32, block_size: u32,
    )
);

cuda_export!(
    pub(crate) LatentExpandedLayoutKernel = "libmir_cuda_latent_expanded_layout"(
        contexts: &DeviceBuffer<u32>, regions: &mut DeviceBuffer<u32>,
        table: &mut DeviceBuffer<u32>, sequences: u32, stride: u32,
    )
);

cuda_export!(
    pub(crate) LatentExpandFillKernel = "libmir_cuda_latent_expand_fill_bf16"(
        latent: &DeviceBuffer<bf16>, keys: &mut DeviceBuffer<bf16>,
        values: &mut DeviceBuffer<bf16>, rows: u32, heads: u32, nope_dim: u32, value_dim: u32,
        head_dim: u32,
    )
);

cuda_export!(
    pub(crate) LatentExpandedQueryKernel = "libmir_cuda_latent_expanded_query_bf16"(
        query: &DeviceBuffer<bf16>, absorbed: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>, tokens: u32,
        heads: u32, nope_dim: u32, head_dim: u32, temperature_beta: f32,
        temperature_window: u32,
    )
);

cuda_export!(
    pub(crate) LatentExpandedOutputKernel = "libmir_cuda_latent_expanded_output_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32, heads: u32,
        value_dim: u32, head_dim: u32,
    )
);

cuda_export!(
    pub(crate) LatentExpandedStartsKernel = "libmir_cuda_latent_expanded_starts"(
        starts: &mut DeviceBuffer<u32>, tokens: u32, context: u32,
    )
);

/// The kernel that attends every query row over its split of the cache.
#[derive(Clone, Debug)]
pub(super) enum LatentSplit {
    /// A warp per head, eight heads per block.
    Warp(TypedKernel<LatentSplitKernel>),
    /// Sixteen heads of one token per block on tensor cores.
    TensorCore(TypedKernel<LatentMmaKernel>),
    /// Two such row groups per block sharing each key tile; one split only.
    TensorCorePairs(TypedKernel<LatentMmaPairKernel>),
}

/// The kernels of expanded prefill attention.
#[derive(Clone, Debug)]
pub(super) struct ExpandedKernels {
    pub(super) gather: TypedKernel<LatentGatherKernel>,
    pub(super) fill: TypedKernel<LatentExpandFillKernel>,
    pub(super) query: TypedKernel<LatentExpandedQueryKernel>,
    pub(super) output: TypedKernel<LatentExpandedOutputKernel>,
    pub(super) starts: TypedKernel<LatentExpandedStartsKernel>,
    pub(super) layout: TypedKernel<LatentExpandedLayoutKernel>,
}

impl ExpandedKernels {
    pub(super) fn load(module: &Module) -> Result<Self> {
        Ok(Self {
            gather: module.kernel()?,
            fill: module.kernel()?,
            query: module.kernel()?,
            output: module.kernel()?,
            starts: module.kernel()?,
            layout: module.kernel()?,
        })
    }
}
