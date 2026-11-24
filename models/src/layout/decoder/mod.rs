mod attention;
mod defaults;
mod experts;
mod features;
mod latent;
mod parse;
mod pattern;
mod rope;
mod state_space;

pub use attention::AttentionLayerType;
pub use experts::RoutedExpertsConfig;
pub use features::{
    AttentionOutput, LINEAR_ATTENTION_LAYOUT, LINEAR_ATTENTION_LAYOUT_LLAMA_CPP,
    LinearAttentionConfig, LinearAttentionLayout, PerLayerInputConfig, RotaryEmbeddingLayout,
};
pub use latent::{LatentAttentionConfig, LatentDecoderConfig, RotaryPairing};
pub use parse::{QUERY_KEY_ROWS, QUERY_KEY_ROWS_INTERLEAVED};
pub use pattern::AttentionPattern;
pub use rope::{RopeFactors, RopeScaling};
pub use state_space::{
    HybridBlockKind, HybridBlockPattern, HybridFeedForward, HybridLayer, HybridLayerBlock,
    LatentRoutedConfig, Mamba2Config, PredictionModules, STATE_SPACE_LAYOUT,
    STATE_SPACE_LAYOUT_LLAMA_CPP, StateSpaceHybridConfig, StateSpaceLayout, StateSpaceStateDType,
    TimeStepBound, TimeStepLimit,
};

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
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
    /// Sliding layers rotate without `rope_scaling` (Gemma 3).
    pub sliding_attention_rope_unscaled: bool,
    pub full_attention_rope_type: Option<crate::semantic::RotaryAlgorithm>,
    pub sliding_attention_rope_type: Option<crate::semantic::RotaryAlgorithm>,
    pub full_attention_partial_rotary_factor: Option<f64>,
    pub sliding_attention_partial_rotary_factor: Option<f64>,
    pub layer_types: Vec<AttentionLayerType>,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub attention_scale: Option<f64>,
    /// Explicit or built-in normalization; absent means tensor-based discovery.
    pub qk_normalization: Option<crate::semantic::QkNormalization>,
    pub attention_output: AttentionOutput,
    pub sliding_window: Option<usize>,
    pub linear_attention: Option<LinearAttentionConfig>,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub hidden_activation: Option<String>,
    pub final_logit_softcapping: Option<f64>,
    /// Cap on attention scores before the softmax (Gemma 2).
    pub attn_logit_softcapping: Option<f64>,
    /// Factor applied to token embeddings, when the configuration declares one.
    pub embedding_multiplier: Option<f64>,
    /// The decoder's norm: RMS, or mean-centred `LayerNorm` without bias
    /// (Cohere).
    pub norm_kind: crate::layout::NormKind,
    /// How routed experts are selected and weighed: the `DeepSeek` layout's
    /// declared router, else the schema's.
    pub expert_router: crate::semantic::RouterNormalization,
    /// Routed experts on every `step`-th layer only, the others dense at
    /// `dense_intermediate_size` (Llama 4's `interleave_moe_layer_step` and
    /// `intermediate_size_mlp`); `None` routes every layer with experts.
    pub routed_layer_step: Option<std::num::NonZeroUsize>,
    pub dense_intermediate_size: Option<usize>,
    /// Attention and the feed-forward read one normalized input in
    /// parallel and join the residual together (Cohere).
    pub parallel_block: bool,
    /// Per-layer attention traits beyond the layer type.
    pub attention_pattern: AttentionPattern,
    /// Factor on each residual branch before it joins the residual stream
    /// (Granite `residual_multiplier`).
    pub residual_multiplier: Option<f64>,
    /// Divisor of the output logits (Granite `logits_scaling`).
    pub logits_divisor: Option<f64>,
    pub attention_bias: bool,
    pub attention_sinks: bool,
    pub swiglu_limit: Option<f64>,
    /// Query and key projection rows interleave each head's rotary halves
    /// pairwise, as llama.cpp's converter writes Llama-family GGUF files;
    /// loaders restore the `rotate_half` order.
    pub query_key_rows_interleaved: bool,
    pub initial_context_length: Option<usize>,
    /// The last layers reuse the keys and values of the last earlier layer
    /// of their attention type (Gemma 4's E models' `num_kv_shared_layers`).
    pub kv_shared_layers: usize,
    /// Per-layer input embeddings (Gemma 4's E models).
    pub per_layer_input: Option<PerLayerInputConfig>,
    pub state_space: Option<StateSpaceHybridConfig>,
    /// Multi-head latent attention.
    pub latent: Option<LatentDecoderConfig>,
    /// Multi-token prediction modules after the decoder (Qwen3.5's
    /// `mtp_num_hidden_layers`, `num_nextn_predict_layers`); state-space
    /// hybrids declare theirs in [`StateSpaceHybridConfig`].
    pub prediction_depth: Option<std::num::NonZeroUsize>,
    /// The `DeepSeek` expert layout: leading dense layers, routed and shared
    /// experts, and their router.
    pub routed_experts: Option<RoutedExpertsConfig>,
}

#[cfg(test)]
#[allow(clippy::self_named_module_files)]
mod tests;
