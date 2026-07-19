mod array;
mod attention;
mod compiled;
mod decode_graph;
mod decoder;
mod decoder_cache;
mod dense_linear;
pub mod dense_swiglu;
mod embedding;
mod error;
pub(crate) mod expert_fusion;
mod fused_attention;
mod fused_expert_gate_up;
mod fused_gate_up;
mod fused_key_value;
mod gated_delta;
mod gated_delta_layer;
mod gated_full_attention;
mod graph;
pub mod hybrid_linear_moe;
pub mod hybrid_moe;
mod kernels;
mod kv;
mod layer_norm;
mod linear;
mod memory;
mod metadata;
mod model;
mod moe;
mod norm;
mod quantized;
mod sampling;
mod scalar;
mod shared_expert_moe;
mod snapshot;
mod stream;
mod tensors;
#[cfg(test)]
mod tests;
pub mod vision;

pub use array::Array;
pub use attention::{ImageTokenSpan, PagedAttention, RopeOptions, prefix_attention_mask};
pub(crate) use decoder::DecoderModel;
pub use decoder_cache::DecoderCache;
pub use dense_linear::DenseLinear;
pub use embedding::QuantizedEmbedding;
pub use error::{Error, Result};
pub(crate) use expert_fusion::{ExpertFusion, ExpertFusionDecision, configure_expert_fusion};
pub(crate) use fused_attention::FusedAttention;
pub(crate) use fused_expert_gate_up::FusedExpertGateUp;
pub(crate) use fused_gate_up::FusedGateUp;
pub(crate) use fused_key_value::FusedKeyValue;
pub use gated_delta::{GatedDeltaInputs, GatedDeltaState};
pub use gated_delta_layer::{GatedDeltaLayer, GatedDeltaLayerConfig};
pub use gated_full_attention::{GatedFullAttention, GatedFullAttentionConfig};
pub use kv::{KvCache, KvContext, PagedKvContext};
pub(crate) use kv::{
    NATIVE_PAGED_ATTENTION_MIN_CONTEXT, PagedContextMode, native_paged_attention_mode,
    paged_attention_enabled, paged_attention_min_context,
};
pub use layer_norm::LayerNorm;
pub use linear::QuantizedLinear;
pub(crate) use memory::{MemoryStats, configure_recommended_wired_limit, memory_stats};
pub use metadata::Dtype;
pub use model::ModelTensors;
pub use moe::{RouterOutput, SortedExpertInputs};
pub(crate) use norm::NormWeight;
pub use quantized::QuantizedArrays;
pub use sampling::TopK;
pub(crate) use sampling::{DeviceSampling, sample, sample_u32};
pub use shared_expert_moe::{SharedExpertMoe, SharedExpertMoeConfig};
pub use stream::Stream;
pub use tensors::TensorFile;
pub use vision::{pooled::PooledVisionTower, spatial_merge::SpatialMergeVisionTower};

pub fn version() -> Result<String> {
    Ok(mirtal::version()?)
}

pub fn clear_memory_cache() -> Result<()> {
    Ok(mirtal::memory::clear_cache()?)
}
