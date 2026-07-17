mod attention;
mod features;
mod parse;
mod rope;

pub use attention::AttentionLayerType;
pub use features::{AttentionOutput, LinearAttentionConfig, RotaryEmbeddingLayout};
pub use rope::RopeScaling;

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: Option<usize>,
    pub num_global_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<RopeScaling>,
    pub partial_rotary_factor: Option<f64>,
    pub rope_layout: RotaryEmbeddingLayout,
    pub full_attention_rope_theta: Option<f64>,
    pub sliding_attention_rope_theta: Option<f64>,
    pub full_attention_rope_type: Option<String>,
    pub sliding_attention_rope_type: Option<String>,
    pub full_attention_partial_rotary_factor: Option<f64>,
    pub sliding_attention_partial_rotary_factor: Option<f64>,
    pub layer_types: Vec<AttentionLayerType>,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub attention_scale: Option<f64>,
    pub attention_output: AttentionOutput,
    pub sliding_window: Option<usize>,
    pub linear_attention: Option<LinearAttentionConfig>,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub hidden_activation: Option<String>,
    pub final_logit_softcapping: Option<f64>,
}

#[cfg(test)]
mod tests;
