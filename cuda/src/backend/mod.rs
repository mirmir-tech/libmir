mod attention;
mod block;
mod decoder;
mod dense;
mod embedding;
mod init;
mod kv;
mod linear;
mod model;
mod output;
mod planning;
mod router;
mod runtime;
mod sampling;

use std::sync::Arc;

use ::runtime::kv::KvStorageSpec;
pub use attention::{
    BatchedDecodeAttentionBf16, CapturedDecodeAttentionBf16, DecodeAttentionBf16,
    DecodeAttentionConfig, DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights,
    PrefillAttentionBf16,
};
pub use block::{
    BatchedDecodeMoeBlockBf16, BatchedDecodeMoeLayer, CapturedDecodeMoeBlockBf16,
    DecodeGraphAction, DecodeMoeBlockBf16, DecodeMoeBlockConfig, DecodeMoeBlockExecutor,
    DecodeMoeBlockWeights, DecodeMoeLayerTemplate, PrefillMoeBlockBf16,
};
pub use decoder::{RmsNormBf16, RopeBf16};
pub use dense::{
    DecodeDenseSwiGlu, DenseDownSource, DenseDownWeight, DenseGateUpSource, DenseGateUpWeights,
    DenseOutputSource, DenseQkvSource, DenseSwiGluConfig, DenseSwiGluLayerTemplate,
    DenseSwiGluWeights, DenseWeightSource, PrefillDenseSwiGlu,
};
pub use embedding::Bf16Embedding;
pub use kv::{BatchedPagedAttentionBf16, PagedAttentionBf16, PagedDecodeBatch, PagedKvCache};
pub use linear::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    AffineQuantizedPairTensors, AffineQuantizedTensors, Bf16Linear, Bf16LinearPack,
    Bf16LinearPackWeights, Bf16LinearPair, Bf16LinearPairWeights, Bf16Projection, Bf16VectorLinear,
    BlockFp8LinearWeight, BucketedNvFp4MoeBf16, DirectNvFp4MoeBf16, Fp8ResidualLinearWeight,
    GatedActivation, GroupedNvFp4MoeBf16, HybridNvFp4MoeBf16, NvFp4Bf16Linear, NvFp4Bf16Pack,
    NvFp4Config, NvFp4ExpertBank, NvFp4ExpertBankConfig, NvFp4ExpertSource, NvFp4LinearWeight,
    NvFp4Tensors, ProjectionFormat, SelectedAffineGatedBf16Linear, SelectedAffinePairBf16Linear,
    SelectedAffineReduceBf16Linear, SelectedNvFp4LinearBf16, SelectedNvFp4MoeBf16,
    SelectedNvFp4TensorCoreMoeBf16,
};
use mircuda::{Compiler, Context, DeviceInfo, MemoryPool, Stream};
pub use model::{
    CudaDecodeBatch, CudaModelSessionConfig, CudaMoeModelSession, CudaMoeModelTemplate,
};
pub use output::CudaOutputHead;
pub use planning::{
    AttentionExecution, AttentionPlan, AttentionPlanRequest, CudaAttentionPolicy,
    CudaDenseVectorPolicy, CudaDenseWeightPolicy, CudaExecutionPlanner, CudaHardwareProfile,
    CudaKernelAdmission, CudaMemoryArchitecture, CudaMoeBatchPolicy, CudaMoeFusionPolicy,
    CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy, DenseExecution, DensePlan,
    DensePlanRequest, DenseRole, ExecutionPhase, MoeExecution, MoePlan, MoePlanRequest,
    MoeQuantization, OutputHeadExecution, OutputHeadPlan, OutputHeadPlanRequest, PlanSource,
};
pub use router::{RouterBf16, RouterSelection, RouterTensors};
pub use sampling::{DeviceBatchSamplerBf16, DeviceSamplerBf16};

use crate::{Result, kernels::RopeSpec};

/// Initialized native CUDA backend resources.
#[derive(Clone, Debug)]
pub struct CudaBackend {
    inner: Arc<CudaRuntime>,
}

#[derive(Debug)]
struct CudaRuntime {
    device: DeviceInfo,
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    compiler: Compiler,
    planner: CudaExecutionPlanner,
}

impl CudaBackend {
    /// Prepares a cached BF16 linear operation using checkpoint weight layout
    /// `[out, in]`.
    pub fn prepare_bf16_linear(
        &self,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Bf16Linear> {
        Bf16Linear::new(self, tokens, input_features, output_features)
    }

    /// Prepares a native BF16 matrix-vector operation for one decode row.
    pub fn prepare_bf16_vector_linear(
        &self,
        input_features: usize,
        output_features: usize,
    ) -> Result<Bf16VectorLinear> {
        Bf16VectorLinear::new(self, input_features, output_features)
    }

    /// Prepares fixed-shape BF16 RMS normalization.
    pub fn prepare_rms_norm_bf16(
        &self,
        rows: usize,
        features: usize,
        epsilon: f32,
    ) -> Result<RmsNormBf16> {
        RmsNormBf16::new(self, rows, features, epsilon)
    }

    /// Prepares standard half-split, optionally partial `RoPE`.
    pub fn prepare_rope_bf16(&self, spec: RopeSpec) -> Result<RopeBf16> {
        RopeBf16::new(self, spec)
    }

    /// Allocates encoded K/V pages addressed by runtime-owned block
    /// identifiers.
    pub fn prepare_paged_kv(&self, layer: usize, storage: KvStorageSpec) -> Result<PagedKvCache> {
        PagedKvCache::new(self, layer, storage)
    }

    /// Prepares decode GQA over a paged BF16 K/V cache.
    pub fn prepare_paged_attention_bf16(
        &self,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
    ) -> Result<PagedAttentionBf16> {
        PagedAttentionBf16::new(self, cache, query_heads, max_blocks)
    }

    /// Prepares one decode-attention launch spanning independent sequences.
    pub fn prepare_batched_paged_attention_bf16(
        &self,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<BatchedPagedAttentionBf16> {
        BatchedPagedAttentionBf16::new(self, cache, query_heads, max_blocks, max_batch)
    }

    /// Allocates shared block-table metadata for a fixed-capacity microbatch.
    pub fn prepare_paged_decode_batch(
        &self,
        storage: KvStorageSpec,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<PagedDecodeBatch> {
        PagedDecodeBatch::new(self, storage, max_blocks, max_batch)
    }

    /// Materializes one NVFP4 checkpoint weight as the BF16 quality reference.
    pub fn prepare_nvfp4_bf16_linear(
        &self,
        tokens: usize,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<NvFp4Bf16Linear> {
        NvFp4Bf16Linear::new(self, tokens, config, tensors)
    }

    /// Prepares one allocation-free BF16-input affine grouped quantized GEMV.
    pub fn prepare_affine_quantized_bf16_linear(
        &self,
        input_features: usize,
        output_features: usize,
        matrices: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<AffineQuantizedBf16Linear> {
        AffineQuantizedBf16Linear::new(
            self, input_features, output_features, matrices, group_size, bits,
        )
    }

    /// Prepares one tiled BF16-input affine quantized prefill projection.
    pub fn prepare_affine_quantized_bf16_qmm(
        &self,
        tokens: usize,
        config: AffineQuantizedConfig,
        matrices: usize,
    ) -> Result<AffineQuantizedBf16Qmm> {
        AffineQuantizedBf16Qmm::new(self, tokens, config, matrices)
    }

    /// Prepares paired projections for a device-selected expert set.
    pub fn prepare_selected_affine_pair_bf16_linear(
        &self,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<SelectedAffinePairBf16Linear> {
        SelectedAffinePairBf16Linear::new(self, config.spec()?, expert_count, selected_count)
    }

    /// Prepares selected gate/up projections fused with configured activation.
    pub fn prepare_selected_affine_gated_bf16_linear(
        &self,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
        activation: GatedActivation,
    ) -> Result<SelectedAffineGatedBf16Linear> {
        SelectedAffineGatedBf16Linear::new(
            self,
            config.spec()?,
            expert_count,
            selected_count,
            activation,
        )
    }

    /// Prepares selected down projections fused with router-weighted reduction.
    pub fn prepare_selected_affine_reduce_bf16_linear(
        &self,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<SelectedAffineReduceBf16Linear> {
        SelectedAffineReduceBf16Linear::new(self, config.spec()?, expert_count, selected_count)
    }

    /// Explicitly waits for all queued backend work.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.stream.synchronize()?)
    }

    /// Synchronizes and releases unused device-pool storage.
    pub fn trim_memory_pool(&self, retain_bytes: usize) -> Result<()> {
        self.synchronize()?;
        Ok(self.inner.pool.trim_to(retain_bytes)?)
    }
}
