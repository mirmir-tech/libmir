mod array;
mod attention;
mod decode;
pub(crate) mod expert;
mod fused;
pub mod hybrid;
#[cfg(test)]
pub(crate) use attention::tuning::batch::BatchAttentionExecution;
mod binding;
pub mod clamped_routed;
mod compiled;
mod decision;
mod decoder;
pub mod dense;
pub(crate) mod diagnostics;
mod embedding;
mod error;
mod fusion_planner;
mod gate_up_tuning;
mod gated_delta;
mod gated_full_attention;
mod graph;
mod kernels;
mod kv;
#[cfg(test)]
pub(crate) use kv::history as persistent_history;
mod layer_norm;
mod linear;
pub mod lowering;
mod memory;
mod metadata;
mod model;
mod moe;
mod norm;
mod quantized;
mod route_tuning;
mod sampling;
mod scalar;
mod sequence_scoring;
mod shared_expert_moe;
mod snapshot;
mod stream;
mod tensors;
#[cfg(test)]
mod tests;
mod text_embedding;
pub mod vision;

pub use array::Array;
pub use attention::{ImageTokenSpan, PagedAttention, RopeOptions, prefix_attention_mask};
pub use decision::MetalDecisionModel;
pub(crate) use decode::plan_tuning::{
    DecodePlan, DecodePlanAction, DecodePlanKey, context_bucket as decode_context_bucket,
};
pub use decoder::DecoderCache;
pub(crate) use decoder::DecoderModel;
#[cfg(test)]
pub(crate) use decoder::{DecoderExecution, probe};
pub use dense::{DenseEmbedding, DenseLinear};
pub use embedding::QuantizedEmbedding;
pub use error::{Error, Result};
pub(crate) use expert::fusion::{ExpertFusion, ExpertFusionDecision, configure_expert_fusion};
pub(crate) use fused::{
    attention::FusedAttention, expert_gate_up::FusedExpertGateUp, gate_up::FusedGateUp,
    key_value::FusedKeyValue,
};
pub use gated_delta::{GatedDeltaInputs, GatedDeltaLayer, GatedDeltaLayerConfig, GatedDeltaState};
pub use gated_full_attention::{GatedFullAttention, GatedFullAttentionConfig};
pub(crate) use kv::{
    DecodeCapacity, KvPageFormat, NATIVE_PAGED_ATTENTION_MIN_CONTEXT, PagedArenaPool,
    PagedContextMode, native_paged_attention_mode, paged_attention_enabled,
    paged_attention_min_context,
};
pub use kv::{KvCache, KvContext, PagedKvContext};
pub use layer_norm::LayerNorm;
pub use linear::QuantizedLinear;
pub(crate) use memory::{MemoryStats, configure_recommended_wired_limit, memory_stats};
pub use metadata::Dtype;
pub use model::ModelTensors;
pub use moe::{RouterOutput, SortedExpertInputs};
pub(crate) use norm::NormWeight;
pub use quantized::QuantizedArrays;
pub use sampling::TopK;
pub(crate) use sampling::{DeviceSampling, MaskedSampling, TokenAllowlist, sample, sample_u32};
pub use sequence_scoring::SequenceScoringModel;
pub use shared_expert_moe::{SharedExpertMoe, SharedExpertMoeConfig};
pub use stream::Stream;
pub use tensors::TensorFile;
pub use text_embedding::TextEmbeddingModel;
pub use vision::{pooled::PooledVisionTower, spatial_merge::SpatialMergeVisionTower};

pub fn version() -> Result<String> {
    Ok(mirtal::version()?)
}

pub fn clear_memory_cache() -> Result<()> {
    Ok(mirtal::memory::clear_cache()?)
}
