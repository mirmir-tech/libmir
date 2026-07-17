use foundation::model::{ModelFamily, Quantization};
use serde::{Deserialize, Serialize};

use crate::{
    backend::BackendInfo,
    kv::{KvCacheDType, KvQuantMode, KvScaleGranularity},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrace {
    pub model: TraceModel,
    pub backend: BackendInfo,
    pub acceleration: Vec<String>,
    pub decoder: TraceDecoder,
    pub tokenizer: TraceTokenizer,
    pub tensors: TraceTensors,
    pub weights: TraceWeights,
    pub kv_cache: TraceKvCache,
    pub actions: Vec<TraceAction>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceModel {
    pub id: String,
    pub root: String,
    pub family: ModelFamily,
    pub model_type: Option<String>,
    pub dtype: Option<String>,
    pub architectures: Vec<String>,
    pub context_len: usize,
    pub quantization: Quantization,
    pub quantization_group_size: Option<usize>,
    pub quantization_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceDecoder {
    pub layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: Option<usize>,
    pub global_kv_heads: Option<usize>,
    pub full_attention_layers: usize,
    pub sliding_attention_layers: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f64>,
    pub full_attention_rope_theta: Option<f64>,
    pub sliding_attention_rope_theta: Option<f64>,
    pub sliding_window: Option<usize>,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub hidden_activation: Option<String>,
    pub final_logit_softcapping: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceTensors {
    pub tensor_count: usize,
    pub native_tensor_count: usize,
    pub weight_files: usize,
    pub native_shards: usize,
    pub weight_bytes: u64,
    pub tokenizer: bool,
    pub safetensors_index: bool,
    pub readiness: String,
    pub missing: Vec<String>,
    pub native_dtypes: Vec<TraceDTypeCount>,
    pub finite_validation: TraceFiniteValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceTokenizer {
    pub path: Option<String>,
    pub kind: Option<String>,
    pub vocab_size: Option<usize>,
    pub stop_token_ids: Vec<u32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceDTypeCount {
    pub dtype: String,
    pub tensors: usize,
    pub elements: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFiniteValidation {
    pub mode: String,
    pub checked_tensors: usize,
    pub checked_elements: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceWeights {
    pub token_embeddings: String,
    pub final_norm: String,
    pub output_head: String,
    pub output_tied: bool,
    pub layer_count: usize,
    pub attention_layout: String,
    pub mlp_layout: String,
    pub linear_bias_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceKvCache {
    pub dtype: KvCacheDType,
    pub quant_mode: KvQuantMode,
    pub scale_granularity: KvScaleGranularity,
    pub decode_attention: String,
    pub block_size: Option<usize>,
    pub physical_page_key: String,
    pub prefix_cache: bool,
    pub paged_attention: bool,
    pub paged_attention_min_context: Option<usize>,
    pub entry_count: usize,
    pub cached_tokens: usize,
    pub resident_token_slots: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceAction {
    pub stage: String,
    pub detail: String,
}

impl TraceAction {
    #[must_use]
    pub fn new(stage: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            stage: stage.into(),
            detail: detail.into(),
        }
    }
}

impl ModelTrace {
    #[must_use]
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!("model: {} ({:?})", self.model.id, self.model.family),
            format!("backend: {} on {}", self.backend.name, self.backend.device),
            format!(
                "decoder: {} layers, hidden {}, vocab {}, heads {}/{}, experts {}",
                self.decoder.layers,
                self.decoder.hidden_size,
                self.decoder.vocab_size,
                self.decoder.attention_heads,
                self.decoder.kv_heads,
                self.decoder
                    .num_experts
                    .map_or_else(|| "none".into(), |experts| experts.to_string())
            ),
            format!(
                "tokenizer: {}, vocab {}, stops {}",
                self.tokenizer.kind.as_deref().unwrap_or("unavailable"),
                self.tokenizer
                    .vocab_size
                    .map_or_else(|| "unknown".into(), |vocab| vocab.to_string()),
                self.tokenizer.stop_token_ids.len()
            ),
            format!(
                "weights: attention {}, mlp {}, biases {}",
                self.weights.attention_layout,
                self.weights.mlp_layout,
                self.weights.linear_bias_count
            ),
            format!("acceleration: {}", self.acceleration.join("; ")),
            format!(
                "kv_cache: dtype {}, {:?}, decode attention {}, pages keyed by {}",
                self.kv_cache.dtype,
                self.kv_cache.quant_mode,
                self.kv_cache.decode_attention,
                self.kv_cache.physical_page_key
            ),
            format!("readiness: {}", self.tensors.readiness),
        ];
        lines.extend(
            self.actions
                .iter()
                .map(|action| format!("trace.{}: {}", action.stage, action.detail)),
        );
        lines.extend(self.warnings.iter().map(|warning| format!("warning: {warning}")));
        lines
    }
}
