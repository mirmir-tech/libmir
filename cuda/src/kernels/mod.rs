//! CUDA inference kernels owned by libmir.
//!
//! Mircuda supplies compilation and typed dispatch; this module owns model
//! mathematics, quantization formats, launch geometry, and numerical policy.

mod affine;
mod awq;
mod bitsandbytes;
mod clamped_routed;
mod decoder;
mod dense_cast;
mod direct_fp8;
mod elementwise;
mod embedding;
mod encoder;
mod gated_attention;
mod gated_delta;
mod geometry;
mod gptq;
mod linear_fp8;
mod logit_softcap;
mod mrope;
mod mxfp4;
mod nvfp4;
mod output_fp8;
mod packed_gated;
mod packed_int8;
mod paged;
mod qkv;
mod qmm;
mod rms_norm_shift;
mod router;
mod row;
mod sampling;
mod selected;
mod sigmoid;
#[cfg(test)]
mod tests;
mod text;
mod vision;

pub use affine::{
    AffineEmbedding, AffineEmbeddingSpec, AffineGemvLaunch, AffineGemvSpec, AffineQuantizedGemv,
};
pub use awq::{AwqLaunch, AwqLinear, AwqSpec};
pub use bitsandbytes::{BitsAndBytes4BitLaunch, BitsAndBytes4BitLinear, BitsAndBytes4BitSpec};
pub(crate) use clamped_routed::{
    ClampedRoutedAttention, ClampedRoutedBatchSplitDecode, ClampedRoutedKernels, ClampedRoutedSpec,
    ClampedRoutedSplitDecode,
};
pub use decoder::{RmsNorm, RmsNormUnit, Rope, RopeSpec};
pub(crate) use dense_cast::DenseCast;
pub(crate) use direct_fp8::{DirectE5M2WeightOnlyTensorCoreLinear, DirectFp8TensorCoreLinear};
pub use direct_fp8::{
    DirectFp8Activation, DirectFp8Embedding, DirectFp8EmbeddingBatch, DirectFp8EmbeddingSpec,
    DirectFp8Format, DirectFp8Linear, DirectFp8Scale, DirectFp8Scales, DirectFp8Spec,
};
pub use elementwise::ElementwiseBf16;
pub use embedding::Embedding;
pub use encoder::{
    EncoderAttentionF16, EncoderAttentionSpec, EncoderElementwiseF16, EncoderElementwiseSpec,
};
pub use gated_attention::GatedAttentionSplit;
pub use gated_delta::{
    GatedDeltaBatchConvolution, GatedDeltaBatchConvolutionSpec, GatedDeltaBatchRecurrence,
    GatedDeltaBatchSpec, GatedDeltaConvolution, GatedDeltaConvolutionSpec,
    GatedDeltaInputs as GatedDeltaKernelInputs, GatedDeltaLaunch, GatedDeltaRecurrence,
    GatedDeltaSpec, GatedDeltaTransformSpec, GatedDeltaTransforms,
};
pub use gptq::{GptqLaunch, GptqLinear, GptqSpec};
pub use linear_fp8::{BlockFp8LinearKernels, BlockFp8LinearSpec};
pub(crate) use logit_softcap::LogitSoftcap;
pub use mrope::{Mrope, MropeSpec};
pub use mxfp4::{
    MxFp4Embedding, MxFp4EmbeddingOperands, MxFp4EmbeddingSpec, MxFp4GatheredLinear,
    MxFp4GatheredOperands, MxFp4GatheredSpec, MxFp4Linear, MxFp4Spec,
};
pub use nvfp4::{
    BankScaleGeometry, BucketGeometry, BucketQuantize, GroupedQuantize, NvFp4BucketPreparation,
    NvFp4Dequant, NvFp4DequantLaunch, NvFp4Gated, NvFp4GroupedPreparation, NvFp4MicroBanks,
    NvFp4MicroDownKernels, NvFp4MicroDownLaunch, NvFp4MicroDownWorkspace, NvFp4MicroGateLaunch,
    NvFp4MicroGateWorkspace, NvFp4MicroKernels, NvFp4MicroLaunch, NvFp4MicroSpec,
    NvFp4MicroWorkspace, NvFp4Preparation, NvFp4RmsNorm, NvFp4SelectedWeightLaunch,
    NvFp4SelectedWeightPreparation, NvFp4Spec, NvFp4WeightOnly, NvFp4WeightOnlyLaunch,
    NvFp4WeightOnlyTensorCore, scale_elements,
};
pub use output_fp8::{
    Fp8OutputKernels, Fp8OutputSpec, Fp8RefinementKernels, Fp8ResidualWeightBuffers,
};
pub(crate) use packed_gated::PackedGatedBf16;
pub use packed_int8::{
    PackedInt8Embedding, PackedInt8EmbeddingLaunch, PackedInt8EmbeddingSpec, PackedInt8Launch,
    PackedInt8Linear, PackedInt8Spec,
};
pub(crate) use paged::{
    AttentionKernel, BatchedPagedPrefillAttention, KvStoreKernel, MergeAttentionArguments,
    SplitAttentionArguments, SplitAttentionConfigs, SplitAttentionKernels, SplitAttentionNodes,
};
pub use paged::{
    BatchedPagedAttention, BatchedPagedKvGather, BatchedSplitAttentionWorkspace,
    BatchedSplitPagedAttention, PagedAttention, PagedAttentionSpec, PagedKvGather, PagedKvSpec,
    PagedKvStore, PagedPrefillAttention, SplitAttentionWorkspace, SplitPagedAttention,
};
pub(crate) use qkv::{
    BatchedQkvPostprocess, QkvNormalization, QkvPostprocess, QkvPostprocessArguments,
    QkvPostprocessKernel, QkvPostprocessSpec,
};
pub use qmm::{AffineQmmLaunch, AffineQmmSpec, AffineQuantizedQmm};
pub use rms_norm_shift::ShiftedRmsNorm;
pub(crate) use router::{RoutePattern, RoutePatternGenerator, RoutePatternSpec};
pub use router::{RouterSpec, RouterTopK, RouterUnitSpec, RouterUnitTopK};
pub use row::{CopyRowsBf16, GatherRowsBf16, SelectRowBf16};
pub use sampling::{MAX_TOP_K, Sampling, SamplingSpec, SamplingWorkspace};
pub use selected::{
    DenseExpertCanonicalizer, DenseGateUpLayout, DenseGatedActivation, GatedActivation,
    NvFp4BankView, SelectedAffineGated, SelectedAffineGatedLaunch, SelectedAffineGatedSpec,
    SelectedAffinePair, SelectedAffinePairLaunch, SelectedAffinePairSpec, SelectedAffineReduce,
    SelectedAffineReduceLaunch, SelectedAffineReduceSpec, SelectedDenseDispatch,
    SelectedDenseGateLaunch, SelectedDenseMoe, SelectedDenseMoeSpec, SelectedDenseReduceLaunch,
    SelectedNvFp4Gated, SelectedNvFp4Reduce, SelectedNvFp4Spec,
};
pub use sigmoid::{SigmoidElementwiseBf16, SigmoidMultiplyBf16};
pub use text::{L2NormalizeBf16, TextAttention, TextAttentionSpec};
pub(crate) use vision::VisionEmbeddingSplice;
pub use vision::{
    SpatialMergeKernels, VisionAttention, VisionAttentionSpec, VisionClip, VisionClipSpec,
    VisionElementwise, VisionElementwiseSpec, VisionPatchLayout, VisionPool, VisionPoolSpec,
    VisionSpatialRope,
};
