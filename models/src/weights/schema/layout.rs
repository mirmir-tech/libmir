use crate::{
    layout::DecoderConfig,
    weights::{DecoderTensorSchema, TensorCatalog, TensorRequirement},
};

pub(super) fn discover(config: &DecoderConfig, catalog: &TensorCatalog) -> DecoderTensorSchema {
    if uses_fused_qkv_layout(catalog) {
        return fused_qkv_schema(config);
    }
    let mut schema = common_hf_schema(config);
    if config.num_experts.is_some() {
        for layer in 0..config.num_hidden_layers {
            push_routed_moe_layer(&mut schema.requirements, layer);
        }
    } else if uses_qk_norm_layout(catalog) {
        for layer in 0..config.num_hidden_layers {
            push_qk_norm_layer(&mut schema.requirements, layer);
        }
    }
    schema
}

fn common_hf_schema(config: &DecoderConfig) -> DecoderTensorSchema {
    let mut requirements = vec![
        req(
            "token embeddings",
            &[
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "transformer.embedding.word_embeddings.weight",
            ],
        ),
        req(
            "final norm",
            &[
                "model.norm.weight",
                "norm.weight",
                "language_model.model.norm.weight",
                "model.language_model.norm.weight",
                "model.final_layernorm.weight",
            ],
        ),
    ];
    if !config.tie_word_embeddings {
        requirements.push(req(
            "output head",
            &["lm_head.weight", "language_model.lm_head.weight", "embed_out.weight"],
        ));
    }
    for layer in 0..config.num_hidden_layers {
        push_common_layer(&mut requirements, layer, config.attention_k_eq_v);
    }
    DecoderTensorSchema { requirements }
}

fn fused_qkv_schema(config: &DecoderConfig) -> DecoderTensorSchema {
    let mut requirements = vec![
        req(
            "token embeddings",
            &["transformer.embedding.word_embeddings.weight", "model.embed_tokens.weight"],
        ),
        req(
            "final norm",
            &["transformer.encoder.final_layernorm.weight", "model.norm.weight"],
        ),
    ];
    if !config.tie_word_embeddings {
        requirements
            .push(req("output head", &["transformer.output_layer.weight", "lm_head.weight"]));
    }
    for layer in 0..config.num_hidden_layers {
        push_glm_layer(&mut requirements, layer);
    }
    DecoderTensorSchema { requirements }
}

fn uses_fused_qkv_layout(catalog: &TensorCatalog) -> bool {
    catalog.contains("transformer.encoder.layers.0.self_attention.query_key_value.weight")
}

fn uses_qk_norm_layout(catalog: &TensorCatalog) -> bool {
    ["model.layers.0", "layers.0"].into_iter().any(|prefix| {
        catalog.contains(&format!("{prefix}.self_attn.q_norm.weight"))
            && catalog.contains(&format!("{prefix}.self_attn.k_norm.weight"))
    })
}

fn push_common_layer(requirements: &mut Vec<TensorRequirement>, layer: usize, k_eq_v: bool) {
    let prefixes = prefixes(layer);
    let v_requirement = if k_eq_v {
        common_pair(
            "attention v",
            prefixes.refs(),
            "self_attn.v_proj.weight",
            "self_attn.k_proj.weight",
        )
    } else {
        common("attention v", prefixes.refs(), "self_attn.v_proj.weight")
    };
    requirements.extend([
        common("attention q", prefixes.refs(), "self_attn.q_proj.weight"),
        common("attention k", prefixes.refs(), "self_attn.k_proj.weight"),
        v_requirement,
        common("attention o", prefixes.refs(), "self_attn.o_proj.weight"),
        common("mlp gate", prefixes.refs(), "mlp.gate_proj.weight"),
        common("mlp up", prefixes.refs(), "mlp.up_proj.weight"),
        common("mlp down", prefixes.refs(), "mlp.down_proj.weight"),
        common("attention norm", prefixes.refs(), "input_layernorm.weight"),
        common_pair(
            "mlp norm",
            prefixes.refs(),
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
        ),
    ]);
}

fn push_routed_moe_layer(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    let prefixes = prefixes(layer);
    requirements.extend([
        common("attention q norm", prefixes.refs(), "self_attn.q_norm.weight"),
        common("attention k norm", prefixes.refs(), "self_attn.k_norm.weight"),
        common("router", prefixes.refs(), "router.proj.weight"),
        expert("expert gate", prefixes.refs(), "gate_proj"),
        expert("expert up", prefixes.refs(), "up_proj"),
        expert("expert down", prefixes.refs(), "down_proj"),
    ]);
}

fn push_qk_norm_layer(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    requirements.extend([
        common("attention q norm", prefixes(layer).refs(), "self_attn.q_norm.weight"),
        common("attention k norm", prefixes(layer).refs(), "self_attn.k_norm.weight"),
    ]);
}

fn push_glm_layer(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    let prefix = format!("transformer.encoder.layers.{layer}");
    requirements.extend([
        one("fused qkv", format!("{prefix}.self_attention.query_key_value.weight")),
        one("attention dense", format!("{prefix}.self_attention.dense.weight")),
        one("mlp up", format!("{prefix}.mlp.dense_h_to_4h.weight")),
        one("mlp down", format!("{prefix}.mlp.dense_4h_to_h.weight")),
        one("attention norm", format!("{prefix}.input_layernorm.weight")),
        one("mlp norm", format!("{prefix}.post_attention_layernorm.weight")),
    ]);
}

struct Prefixes([String; 4]);

impl Prefixes {
    fn refs(&self) -> [&str; 4] {
        self.0.each_ref().map(String::as_str)
    }
}

fn prefixes(layer: usize) -> Prefixes {
    Prefixes([
        format!("model.layers.{layer}"),
        format!("language_model.model.layers.{layer}"),
        format!("model.language_model.layers.{layer}"),
        format!("layers.{layer}"),
    ])
}

fn one(label: &str, name: String) -> TensorRequirement {
    TensorRequirement::any(label, vec![name])
}

fn common(label: &str, prefixes: [&str; 4], suffix: &str) -> TensorRequirement {
    TensorRequirement::any(label, prefixes.map(|prefix| format!("{prefix}.{suffix}")).to_vec())
}

fn common_pair(label: &str, prefixes: [&str; 4], first: &str, second: &str) -> TensorRequirement {
    let aliases = prefixes
        .into_iter()
        .flat_map(|prefix| [format!("{prefix}.{first}"), format!("{prefix}.{second}")])
        .collect();
    TensorRequirement::any(label, aliases)
}

fn expert(label: &str, prefixes: [&str; 4], projection: &str) -> TensorRequirement {
    let aliases = prefixes
        .into_iter()
        .flat_map(|prefix| {
            [
                format!("{prefix}.experts.switch_glu.{projection}.weight"),
                format!("{prefix}.experts.0.{projection}.weight"),
            ]
        })
        .collect();
    TensorRequirement::any(label, aliases)
}

fn req(label: &str, aliases: &[&str]) -> TensorRequirement {
    TensorRequirement::any(label, aliases.iter().map(ToString::to_string).collect())
}
