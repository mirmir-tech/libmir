mod attention;
mod block;
mod decoder;
mod dense;
mod embedding;
mod gated_delta;
mod gated_full_attention;
mod gated_full_attention_layer;
mod hybrid_layer;
mod hybrid_model;
mod init;
mod kv;
mod linear;
mod model;
mod output;
mod planning;
mod prepare;
mod router;
mod runtime;
mod sampling;
mod shared_moe;
mod task;
mod vision;

use std::sync::Arc;

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
pub use embedding::{AffineQuantizedEmbedding, Bf16Embedding};
pub use gated_delta::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, CudaAffineGatedDeltaExecution,
    CudaAffineGatedDeltaLayer, CudaGatedDeltaState, GatedDeltaInputs, GatedDeltaStateConfig,
};
pub use gated_full_attention::{
    AffineGatedFullAttentionConfig, AffineGatedFullAttentionWeights, CudaAffineGatedFullAttention,
    CudaAffineGatedFullAttentionExecution, CudaAffineGatedFullAttentionState,
};
pub use gated_full_attention_layer::{
    AffineGatedFullAttentionMoeLayerConfig, CudaAffineGatedFullAttentionMoeExecution,
    CudaAffineGatedFullAttentionMoeLayer,
};
pub use hybrid_layer::{
    AffineGatedDeltaMoeLayerConfig, CudaAffineGatedDeltaMoeExecution, CudaAffineGatedDeltaMoeLayer,
};
pub use hybrid_model::{
    CudaHybridLinearLayerState, CudaHybridLinearModelSession, CudaHybridLinearModelTemplate,
};
pub use kv::{BatchedPagedAttentionBf16, PagedAttentionBf16, PagedDecodeBatch, PagedKvCache};
pub use linear::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    AffineQuantizedPairTensors, AffineQuantizedTensors, AffineQuantizedWeight, Bf16Linear,
    Bf16LinearPack, Bf16LinearPackWeights, Bf16LinearPair, Bf16LinearPairWeights, Bf16Projection,
    Bf16VectorLinear, BlockFp8LinearWeight, BucketedNvFp4MoeBf16, DirectNvFp4MoeBf16,
    Fp8ResidualLinearWeight, GatedActivation, GroupedNvFp4MoeBf16, HybridNvFp4MoeBf16,
    NvFp4Bf16Linear, NvFp4Bf16Pack, NvFp4Config, NvFp4ExpertBank, NvFp4ExpertBankConfig,
    NvFp4ExpertSource, NvFp4LinearWeight, NvFp4Tensors, ProjectionFormat,
    SelectedAffineGatedBf16Linear, SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear,
    SelectedNvFp4LinearBf16, SelectedNvFp4MoeBf16, SelectedNvFp4TensorCoreMoeBf16,
};
use mircuda::{Compiler, Context, DeviceInfo, MemoryPool, Stream};
pub use model::{
    CudaDecodeBatch, CudaModelSessionConfig, CudaMoeModelSession, CudaMoeModelTemplate,
};
pub use output::{CudaAffineOutputHead, CudaOutputHead};
pub use planning::{
    AttentionExecution, AttentionPlan, AttentionPlanRequest, CudaAttentionPolicy,
    CudaDenseVectorPolicy, CudaDenseWeightPolicy, CudaExecutionPlanner, CudaHardwareProfile,
    CudaKernelAdmission, CudaMemoryArchitecture, CudaMoeBatchPolicy, CudaMoeFusionPolicy,
    CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy, DenseExecution, DensePlan,
    DensePlanRequest, DenseRole, ExecutionPhase, MoeExecution, MoePlan, MoePlanRequest,
    MoeQuantization, OutputHeadExecution, OutputHeadPlan, OutputHeadPlanRequest, PlanSource,
};
pub use router::{AffineRouterBf16, RouterBf16, RouterSelection, RouterTensors};
pub use sampling::{DeviceBatchSamplerBf16, DeviceSamplerBf16};
pub use shared_moe::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, CudaAffineSharedExpertMoe,
    CudaAffineSharedExpertMoeExecution,
};
pub use task::{CudaSequenceScoringModel, CudaTextEmbeddingModel};
pub use vision::{CudaPooledVisionTower, CudaSpatialMergeVisionTower};

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
