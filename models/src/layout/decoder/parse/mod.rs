use std::path::Path;

use serde_json::Value;

use super::{
    AttentionLayerType, DecoderConfig, attention, defaults, experts, features, latent, rope,
    state_space,
};
use crate::{error::Result, layout::ModelLayout};

mod fields;
mod value;

use fields::{
    attention_scale, decoder_value, hybrid_layer_types, intermediate_size, logits_divisor,
    max_position_embeddings, prediction_depth, query_key_rows_interleaved, resolve_head_dim,
    rms_norm_eps, routed_layer_step, shared_expert_intermediate_size, tie_word_embeddings,
};
pub(super) use value::{
    invalid, optional_bool, optional_f64, optional_string, optional_usize, require_usize,
};
use value::{partial_rotary_factor, rope_theta};

/// The `config.json` key GGUF imports write for Llama-family files, and its
/// one value; Hugging Face checkpoints never carry it.
pub const QUERY_KEY_ROWS: &str = "mir_query_key_rows";
pub const QUERY_KEY_ROWS_INTERLEAVED: &str = "rope_interleaved";

/// The names checkpoints give their routed-expert count and top-k.
const EXPERT_COUNT: &[&str] = &["num_experts", "num_local_experts", "n_routed_experts"];
const EXPERTS_PER_TOKEN: &[&str] = &["top_k_experts", "num_experts_per_tok", "experts_per_token"];

impl DecoderConfig {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let mut config = Self::from_config_path(&layout.config_path)?;
        if let Some(path) = layout.configuration_path.as_deref() {
            let value = crate::layout::json::read(path)?;
            if let Some(context) = optional_usize(
                &value,
                &["max_position_embeddings", "max_sequence_length", "seq_length"],
            )? {
                config.max_position_embeddings = config.max_position_embeddings.max(context);
            }
        }
        Ok(config)
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_value(&crate::layout::json::read(path)?)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let decoder = decoder_value(value);
        let num_attention_heads = require_usize(decoder, &["num_attention_heads", "n_head"])?;
        if num_attention_heads == 0 {
            return Err(invalid("num_attention_heads must be greater than zero"));
        }
        let hidden_size = require_usize(decoder, &["hidden_size", "n_embd"])?;
        let defaults = defaults::for_checkpoint(decoder);
        let state_space = state_space::state_space_hybrid(decoder)?;
        let num_hidden_layers = match &state_space {
            Some(hybrid) => hybrid.blocks.layers().len(),
            None => require_usize(decoder, &["num_hidden_layers", "num_layers", "n_layer"])?,
        };
        let latent = latent::latent_decoder(decoder)?;
        let layer_types = match &state_space {
            Some(hybrid) => hybrid_layer_types(hybrid),
            None => attention::layer_types(decoder, num_hidden_layers)?,
        };
        let geometry = (hidden_size, num_attention_heads);
        let (layer_types, head_dim) =
            attention_geometry(decoder, latent.as_ref(), layer_types, geometry)?;
        let attention_pattern = super::pattern::parse(decoder, &defaults, &layer_types)?;
        let partial_rotary_factor = partial_rotary_factor(decoder)?;
        let layer_partial_rotary = |kind| {
            Ok::<_, crate::error::ModelsError>(
                attention::partial_rotary_factor(decoder, kind)?.or(partial_rotary_factor),
            )
        };
        let (routed_experts, expert_router) =
            experts::routed_experts(decoder, latent.is_some(), &defaults)?;
        let config = Self {
            hidden_size,
            intermediate_size: intermediate_size(decoder)?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: optional_usize(
                decoder,
                &["num_key_value_heads", "multi_query_group_num", "n_kv_heads"],
            )?
            .unwrap_or(num_attention_heads),
            head_dim,
            global_head_dim: optional_usize(decoder, &["global_head_dim"])?,
            num_global_key_value_heads: optional_usize(decoder, &["num_global_key_value_heads"])?,
            vocab_size: require_usize(decoder, &["vocab_size"])?,
            max_position_embeddings: max_position_embeddings(decoder)?,
            rms_norm_eps: rms_norm_eps(decoder)?,
            rope_theta: rope_theta(decoder)?,
            rope_scaling: rope::scaling(decoder)?,
            partial_rotary_factor,
            rope_layout: features::rope_layout(decoder)?,
            full_attention_rope_theta: attention::rope_theta(decoder, "full_attention"),
            // Gemma 3 names its sliding layers' base `rope_local_base_freq`,
            // and they are unscaled (`Gemma3TextModel.rotary_emb_local`).
            sliding_attention_rope_theta: attention::rope_theta(decoder, "sliding_attention")
                .or(optional_f64(decoder, &["rope_local_base_freq"])?),
            sliding_attention_rope_unscaled: decoder.get("rope_local_base_freq").is_some(),
            full_attention_rope_type: attention::rope_type(decoder, "full_attention")?,
            sliding_attention_rope_type: attention::rope_type(decoder, "sliding_attention")?,
            full_attention_partial_rotary_factor: layer_partial_rotary("full_attention")?,
            sliding_attention_partial_rotary_factor: layer_partial_rotary("sliding_attention")?,
            layer_types,
            tie_word_embeddings: tie_word_embeddings(value, decoder)?,
            attention_k_eq_v: optional_bool(decoder, "attention_k_eq_v")?.unwrap_or(false),
            attention_scale: attention_scale(decoder)?.or(defaults.attention_scale),
            qk_normalization: defaults.normalization(decoder)?,
            attention_output: features::attention_output(decoder)?,
            sliding_window: optional_usize(decoder, &["sliding_window"])?,
            linear_attention: features::linear_attention(decoder)?,
            num_experts: optional_usize(decoder, EXPERT_COUNT)?,
            top_k_experts: optional_usize(decoder, EXPERTS_PER_TOKEN)?,
            moe_intermediate_size: optional_usize(decoder, &["moe_intermediate_size"])?,
            shared_expert_intermediate_size: shared_expert_intermediate_size(decoder, &defaults)?,
            hidden_activation: optional_string(decoder, "hidden_activation")?
                .or(optional_string(decoder, "hidden_act")?),
            final_logit_softcapping: optional_f64(decoder, &["final_logit_softcapping"])?,
            attn_logit_softcapping: optional_f64(decoder, &["attn_logit_softcapping"])?,
            embedding_multiplier: optional_f64(decoder, &["embedding_multiplier"])?
                .or(defaults.embedding_multiplier(hidden_size)?),
            norm_kind: defaults.norm_kind(),
            parallel_block: defaults.parallel_block,
            expert_router,
            routed_layer_step: routed_layer_step(decoder)?,
            dense_intermediate_size: optional_usize(decoder, &["intermediate_size_mlp"])?,
            attention_pattern,
            residual_multiplier: optional_f64(decoder, &["residual_multiplier"])?,
            logits_divisor: logits_divisor(decoder)?,
            attention_bias: optional_bool(decoder, "attention_bias")?.unwrap_or(false),
            attention_sinks: optional_bool(decoder, "attention_sinks")?.unwrap_or(false),
            swiglu_limit: optional_f64(decoder, &["swiglu_limit"])?,
            // transformers `Llama4TextRotaryEmbedding` rotates adjacent pairs.
            query_key_rows_interleaved: query_key_rows_interleaved(decoder)?
                || defaults.attention_pattern == defaults::AttentionPatternDefault::Llama4,
            initial_context_length: optional_usize(decoder, &["initial_context_length"])?,
            kv_shared_layers: optional_usize(decoder, &["num_kv_shared_layers"])?.unwrap_or(0),
            per_layer_input: features::per_layer_input(decoder)?,
            prediction_depth: prediction_depth(decoder, state_space.is_some())?,
            state_space,
            routed_experts,
            latent,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.num_key_value_heads == 0 || self.head_dim == 0 {
            return Err(invalid("KV head count and head_dim must be greater than zero"));
        }
        if self.global_head_dim == Some(0) || self.num_global_key_value_heads == Some(0) {
            return Err(invalid("global attention dimensions must be greater than zero"));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(invalid("num_attention_heads must be divisible by num_key_value_heads"));
        }
        if self.sliding_window == Some(0) {
            return Err(invalid("sliding_window must be greater than zero"));
        }
        if (self.num_hidden_layers - self.kv_shared_layers.min(self.num_hidden_layers)
            ..self.num_hidden_layers)
            .any(|index| self.key_value_source(index).is_none())
        {
            return Err(invalid("a K/V-sharing layer has no earlier layer of its type to share"));
        }
        if [self.attention_scale, self.residual_multiplier, self.logits_divisor]
            .into_iter()
            .flatten()
            .any(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(invalid(
                "attention, residual and logit scales must be finite and positive",
            ));
        }
        if self.num_experts == Some(0) || self.top_k_experts == Some(0) {
            return Err(invalid("MoE expert counts must be greater than zero"));
        }
        if let (Some(top_k), Some(experts)) = (self.top_k_experts, self.num_experts)
            && top_k > experts
        {
            return Err(invalid("top_k_experts cannot exceed num_experts"));
        }
        Ok(())
    }
}

/// Layer kinds and query/key head width: latent decoders attend through the
/// latent in every softmax layer, with heads `nope + rope` wide.
fn attention_geometry(
    decoder: &Value,
    latent: Option<&super::LatentDecoderConfig>,
    layers: Vec<AttentionLayerType>,
    (hidden_size, heads): (usize, usize),
) -> Result<(Vec<AttentionLayerType>, usize)> {
    match latent {
        Some(latent) => Ok((latent_layer_types(layers)?, latent.attention.query_head_dim())),
        None => Ok((layers, resolve_head_dim(decoder, hidden_size, heads)?)),
    }
}

/// Every softmax layer of a latent-attention decoder attends through the
/// latent.
fn latent_layer_types(layers: Vec<AttentionLayerType>) -> Result<Vec<AttentionLayerType>> {
    layers
        .into_iter()
        .map(|layer| match layer {
            AttentionLayerType::Full => Ok(AttentionLayerType::Latent),
            AttentionLayerType::Linear => Ok(AttentionLayerType::Linear),
            AttentionLayerType::Sliding
            | AttentionLayerType::StateSpace
            | AttentionLayerType::Latent => {
                Err(invalid("latent attention supports only full-attention layers"))
            },
        })
        .collect()
}
