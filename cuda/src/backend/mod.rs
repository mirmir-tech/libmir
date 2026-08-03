mod attention;
mod block;
mod clamped_routed;
mod decoder;
mod dense;
mod embedding;
mod gated_delta;
mod gated_full_attention;
mod hybrid_layer;
mod init;
mod kv;
mod linear;
mod model;
mod output;
mod planning;
mod prepare;
mod profile;
mod router;
mod runtime;
mod sampling;
mod shared_moe;
mod shared_routed;
mod task;
mod tuning;
mod vision;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

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
pub use clamped_routed::{CudaClampedRoutedModelSession, CudaClampedRoutedModelTemplate};
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
    AffineGatedFullAttentionConfig, AffineGatedFullAttentionMoeLayerConfig,
    AffineGatedFullAttentionWeights, CudaAffineGatedFullAttention,
    CudaAffineGatedFullAttentionExecution, CudaAffineGatedFullAttentionMoeExecution,
    CudaAffineGatedFullAttentionMoeLayer, CudaAffineGatedFullAttentionState,
};
pub use hybrid_layer::{
    AffineGatedDeltaMoeLayerConfig, CudaAffineGatedDeltaMoeExecution, CudaAffineGatedDeltaMoeLayer,
};
pub use kv::{
    AttentionSplitMeasurement, BatchedPagedAttentionBf16, BatchedPrefillPagedAttentionBf16,
    PagedAttentionBf16, PagedDecodeBatch, PagedKvCache, PagedPrefillBatch,
    attention_execution_average, candidate_partitions, sample_contexts, select_attention_execution,
};
pub use linear::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    AffineQuantizedPairTensors, AffineQuantizedTensors, AffineQuantizedWeight, Bf16Linear,
    Bf16LinearPack, Bf16LinearPackWeights, Bf16LinearPair, Bf16LinearPairWeights, Bf16Projection,
    Bf16VectorLinear, Bf16VendorLinear, BlockFp8LinearWeight, BucketedNvFp4MoeBf16,
    CheckpointProjectionWeight, CompressedInt8Bf16Linear, CompressedInt8Weight, DenseExpertWeights,
    DirectFp8Bf16Linear, DirectFp8CheckpointWeight, DirectFp8EmbeddingLookup, DirectNvFp4MoeBf16,
    Fp8ResidualLinearWeight, GatedActivation, GroupedNvFp4MoeBf16, HybridNvFp4MoeBf16,
    MxFp4Bf16Linear, MxFp4CheckpointWeight, MxFp4EmbeddingLookup, MxFp4ExpertWeights,
    MxFp4GatheredBf16Linear, MxFp4GatheredMoeBf16, MxFp8Bf16Linear, MxFp8CheckpointWeight,
    MxFp8EmbeddingLookup, MxFp8ExpertWeights, MxFp8GatheredBf16Linear, MxFp8GatheredMoeBf16,
    NvFp4Bf16Linear, NvFp4Bf16Pack, NvFp4Config, NvFp4ExpertBank, NvFp4ExpertBankConfig,
    NvFp4ExpertSource, NvFp4LinearWeight, NvFp4Tensors, NvFp4WeightOnlyBf16Linear,
    NvFp4WeightOnlyWeight, PackedIntegerBf16Linear, PackedIntegerWeight, ProjectionFormat,
    SelectedAffineGatedBf16Linear, SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear,
    SelectedNvFp4LinearBf16, SelectedNvFp4MoeBf16, SelectedNvFp4TensorCoreMoeBf16,
};
use linear::{BucketedNvFp4Scratch, BucketedNvFp4ScratchConfig};
use mircuda::{Compiler, Context, DeviceInfo, MemoryPool, Stream};
pub use model::{
    CudaDecodeBatch, CudaModelSessionConfig, CudaMoeModelSession, CudaMoeModelTemplate,
};
pub use output::{CudaAffineOutputHead, CudaOutputHead};
pub use planning::{
    AttentionExecution, AttentionPlan, AttentionPlanRequest, CudaAttentionPolicy,
    CudaDenseVectorPolicy, CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaExecutionPlanner,
    CudaHardwareProfile, CudaKernelAdmission, CudaMemoryArchitecture, CudaMoeBatchPolicy,
    CudaMoeFusionPolicy, CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy,
    DenseExecution, DensePlan, DensePlanRequest, DenseRole, ExecutionPhase, MoeExecution, MoePlan,
    MoePlanRequest, MoeQuantization, OutputHeadExecution, OutputHeadPlan, OutputHeadPlanRequest,
    PlanSource,
};
pub use profile::{DeviceTimer, ProfilerCapture};
pub use router::{AffineRouterBf16, RouterBf16, RouterSelection, RouterTensors};
pub use sampling::{DeviceBatchSamplerBf16, DeviceSamplerBf16};
pub use shared_moe::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, CudaAffineSharedExpertMoe,
    CudaAffineSharedExpertMoeExecution,
};
pub(crate) use shared_routed::{
    CudaSharedRoutedDecodeBatch, CudaSharedRoutedPrefillBatch, SharedRoutedCheckpoint,
};
pub use shared_routed::{
    CudaSharedRoutedLayerState, CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate,
};
pub use task::{CudaSequenceScoringModel, CudaTextEmbeddingModel};
pub use tuning::{AttentionFamily, AttentionProfileRequest, CudaTuningConfig, CudaTuningMode};
pub use vision::{CudaPooledVisionTower, CudaSpatialMergeVisionTower};

use crate::{Error, Result};

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
    mxfp8_scratch: Mutex<HashMap<(usize, usize), Arc<mircuda::MxFp8TensorCoreScratch>>>,
    nvfp4_bucket_scratch:
        Mutex<HashMap<BucketedNvFp4ScratchConfig, std::sync::Weak<Mutex<BucketedNvFp4Scratch>>>>,
    planner: CudaExecutionPlanner,
    tuner: tuning::CudaAutoTuner,
}

impl CudaBackend {
    fn mxfp8_tensor_core_scratch(
        &self,
        spec: mircuda::MxFp8Spec,
    ) -> Result<Arc<mircuda::MxFp8TensorCoreScratch>> {
        let key = (spec.tokens(), spec.input_features());
        let mut cache = self
            .inner
            .mxfp8_scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("MXFP8 scratch cache lock is poisoned"))?;
        if let Some(scratch) = cache.get(&key) {
            return Ok(scratch.clone());
        }
        let scratch = Arc::new(mircuda::MxFp8TensorCoreScratch::new(
            &self.inner.context,
            &self.inner.pool,
            &self.inner.stream,
            spec,
        )?);
        cache.insert(key, scratch.clone());
        drop(cache);
        Ok(scratch)
    }
}
