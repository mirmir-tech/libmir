//! CUDA inference kernels owned by libmir.
//!
//! Mircuda supplies compilation and typed dispatch; this module owns model
//! mathematics, quantization formats, launch geometry, and numerical policy.

mod affine;
mod decoder;
mod elementwise;
mod embedding;
mod geometry;
mod linear_fp8;
mod nvfp4;
mod nvfp4_buckets;
mod nvfp4_grouped;
mod nvfp4_micro;
mod nvfp4_selected;
mod output_fp8;
mod packed_gated;
mod paged;
mod qkv;
mod qmm;
mod router;
mod row;
mod sampling;
mod selected;
mod selected_nvfp4;
#[cfg(test)]
mod tests;

pub use affine::{AffineGemvLaunch, AffineGemvSpec, AffineQuantizedGemv};
pub use decoder::{RmsNorm, RmsNormUnit, Rope, RopeSpec};
pub use elementwise::ElementwiseBf16;
pub use embedding::Embedding;
pub use linear_fp8::{BlockFp8LinearKernels, BlockFp8LinearSpec};
pub use nvfp4::{
    NvFp4Dequant, NvFp4DequantLaunch, NvFp4Gated, NvFp4Preparation, NvFp4RmsNorm, NvFp4Spec,
    scale_elements,
};
pub use nvfp4_buckets::{BucketGeometry, BucketQuantize, NvFp4BucketPreparation};
pub use nvfp4_grouped::{BankScaleGeometry, GroupedQuantize, NvFp4GroupedPreparation};
pub use nvfp4_micro::{
    NvFp4MicroBanks, NvFp4MicroDownKernels, NvFp4MicroDownLaunch, NvFp4MicroDownWorkspace,
    NvFp4MicroGateLaunch, NvFp4MicroGateWorkspace, NvFp4MicroKernels, NvFp4MicroLaunch,
    NvFp4MicroSpec, NvFp4MicroWorkspace,
};
pub use nvfp4_selected::{NvFp4SelectedWeightLaunch, NvFp4SelectedWeightPreparation};
pub use output_fp8::{
    Fp8OutputKernels, Fp8OutputSpec, Fp8RefinementKernels, Fp8ResidualWeightBuffers,
};
pub(crate) use packed_gated::PackedGatedBf16;
pub(crate) use paged::{
    AttentionKernel, KvStoreKernel, MergeAttentionArguments, SplitAttentionArguments,
    SplitAttentionConfigs, SplitAttentionKernels, SplitAttentionNodes,
};
pub use paged::{
    BatchedPagedAttention, PagedAttention, PagedAttentionSpec, PagedKvSpec, PagedKvStore,
    PagedPrefillAttention, SplitAttentionWorkspace, SplitPagedAttention,
};
pub(crate) use qkv::{
    BatchedQkvPostprocess, QkvNormalization, QkvPostprocess, QkvPostprocessArguments,
    QkvPostprocessKernel, QkvPostprocessSpec,
};
pub use qmm::{AffineQmmLaunch, AffineQmmSpec, AffineQuantizedQmm};
pub use router::{RouterSpec, RouterTopK};
pub use row::SelectRowBf16;
pub use sampling::{MAX_TOP_K, Sampling, SamplingSpec, SamplingWorkspace};
pub use selected::{
    GatedActivation, SelectedAffineGated, SelectedAffineGatedLaunch, SelectedAffineGatedSpec,
    SelectedAffinePair, SelectedAffinePairLaunch, SelectedAffinePairSpec, SelectedAffineReduce,
    SelectedAffineReduceLaunch, SelectedAffineReduceSpec,
};
pub use selected_nvfp4::{
    NvFp4BankView, SelectedNvFp4Gated, SelectedNvFp4Reduce, SelectedNvFp4Spec,
};
