//! CUDA inference kernels owned by libmir.
//!
//! Mircuda supplies compilation and typed dispatch; this module owns model
//! mathematics, quantization formats, launch geometry, and numerical policy.

mod affine;
mod affine_embedding;
mod decoder;
mod elementwise;
mod embedding;
mod gated_attention;
mod gated_delta;
mod geometry;
mod linear_fp8;
mod mrope;
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
mod rms_norm_shift;
mod router;
mod router_unit;
mod row;
mod sampling;
mod selected;
mod selected_nvfp4;
mod sigmoid;
#[cfg(test)]
mod tests;
mod vision;
mod vision_attention;
mod vision_clip;
mod vision_patch;
mod vision_pooling;
mod vision_spatial_merge;
mod vision_splice;

pub use affine::{AffineGemvLaunch, AffineGemvSpec, AffineQuantizedGemv};
pub use affine_embedding::{AffineEmbedding, AffineEmbeddingSpec};
pub use decoder::{RmsNorm, RmsNormUnit, Rope, RopeSpec};
pub use elementwise::ElementwiseBf16;
pub use embedding::Embedding;
pub use gated_attention::GatedAttentionSplit;
pub use gated_delta::{
    GatedDeltaConvolution, GatedDeltaConvolutionSpec, GatedDeltaLaunch, GatedDeltaRecurrence,
    GatedDeltaSpec, GatedDeltaTransformSpec, GatedDeltaTransforms,
};
pub use linear_fp8::{BlockFp8LinearKernels, BlockFp8LinearSpec};
pub use mrope::{Mrope, MropeSpec};
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
pub use rms_norm_shift::ShiftedRmsNorm;
pub use router::{RouterSpec, RouterTopK};
pub use router_unit::{RouterUnitSpec, RouterUnitTopK};
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
pub use sigmoid::{SigmoidElementwiseBf16, SigmoidMultiplyBf16};
pub use vision::{VisionElementwise, VisionElementwiseSpec};
pub use vision_attention::{VisionAttention, VisionAttentionSpec, VisionSpatialRope};
pub use vision_clip::{VisionClip, VisionClipSpec};
pub use vision_patch::VisionPatchLayout;
pub use vision_pooling::{VisionPool, VisionPoolSpec};
pub use vision_spatial_merge::SpatialMergeKernels;
pub(crate) use vision_splice::VisionEmbeddingSplice;
