use crate::{
    layout::{AttentionLayerType, DecoderConfig},
    weights::{DecoderTensorSchema, TensorCatalog, TensorRequirement},
};

pub(super) fn uses_layout(config: &DecoderConfig, catalog: &TensorCatalog) -> bool {
    config.uses_hybrid_linear_moe_stack()
        && catalog.contains("language_model.model.layers.0.linear_attn.in_proj_qkv.weight")
        && catalog.contains("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
}

pub(super) fn schema(config: &DecoderConfig) -> DecoderTensorSchema {
    let mut requirements = base_requirements(config);
    for layer in 0..config.num_hidden_layers {
        push_common_layer(&mut requirements, layer);
        match config.layer_type(layer) {
            AttentionLayerType::Linear => push_linear_attention(&mut requirements, layer),
            AttentionLayerType::Full => push_full_attention(&mut requirements, layer),
            AttentionLayerType::Sliding => {
                unreachable!("hybrid linear layout has no sliding layers")
            },
        }
    }
    DecoderTensorSchema { requirements }
}

fn base_requirements(config: &DecoderConfig) -> Vec<TensorRequirement> {
    let mut requirements = vec![
        one("token embeddings", "language_model.model.embed_tokens.weight"),
        one("final norm", "language_model.model.norm.weight"),
    ];
    if !config.tie_word_embeddings {
        requirements.push(one("output head", "language_model.lm_head.weight"));
    }
    requirements
}

fn push_common_layer(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    let prefix = format!("language_model.model.layers.{layer}");
    requirements.extend([
        one("attention norm", format!("{prefix}.input_layernorm.weight")),
        one("MLP norm", format!("{prefix}.post_attention_layernorm.weight")),
        one("MoE router", format!("{prefix}.mlp.gate.weight")),
        one("shared expert gate", format!("{prefix}.mlp.shared_expert_gate.weight")),
        one(
            "shared expert gate projection",
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        ),
        one(
            "shared expert up projection",
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        ),
        one(
            "shared expert down projection",
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        ),
        one(
            "routed expert gate projection",
            format!("{prefix}.mlp.switch_mlp.gate_proj.weight"),
        ),
        one("routed expert up projection", format!("{prefix}.mlp.switch_mlp.up_proj.weight")),
        one(
            "routed expert down projection",
            format!("{prefix}.mlp.switch_mlp.down_proj.weight"),
        ),
    ]);
}

fn push_linear_attention(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    let prefix = format!("language_model.model.layers.{layer}.linear_attn");
    requirements.extend([
        one("linear attention A", format!("{prefix}.A_log")),
        one("linear attention convolution", format!("{prefix}.conv1d.weight")),
        one("linear attention time bias", format!("{prefix}.dt_bias")),
        one("linear attention QKV projection", format!("{prefix}.in_proj_qkv.weight")),
        one("linear attention gate projection", format!("{prefix}.in_proj_z.weight")),
        one("linear attention beta projection", format!("{prefix}.in_proj_b.weight")),
        one("linear attention alpha projection", format!("{prefix}.in_proj_a.weight")),
        one("linear attention norm", format!("{prefix}.norm.weight")),
        one("linear attention output projection", format!("{prefix}.out_proj.weight")),
    ]);
}

fn push_full_attention(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    let prefix = format!("language_model.model.layers.{layer}.self_attn");
    requirements.extend([
        one("attention q projection", format!("{prefix}.q_proj.weight")),
        one("attention k projection", format!("{prefix}.k_proj.weight")),
        one("attention v projection", format!("{prefix}.v_proj.weight")),
        one("attention output projection", format!("{prefix}.o_proj.weight")),
        one("attention q norm", format!("{prefix}.q_norm.weight")),
        one("attention k norm", format!("{prefix}.k_norm.weight")),
    ]);
}

fn one(label: &str, name: impl Into<String>) -> TensorRequirement {
    TensorRequirement::any(label, vec![name.into()])
}
