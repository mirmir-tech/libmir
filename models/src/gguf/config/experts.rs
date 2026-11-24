//! The `DeepSeek` expert layout under its Hugging Face names
//! (`n_routed_experts`, `first_k_dense_replace`, `scoring_func`, …), as
//! llama.cpp declares it for `deepseek2` and `glm4moe`.

use serde_json::{Map, Value, json};

use super::{
    super::{GgufHeader, MetadataValue},
    decimal,
};

/// Architectures whose experts follow the `DeepSeek` layout without latent
/// attention.
pub(super) fn declares_routed_experts(architecture: &str) -> bool {
    architecture == "glm4moe"
}

/// Inserts leading dense layers, shared experts, and the router.
pub(super) fn routed_experts(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) {
    let key = |name: &str| format!("{architecture}.{name}");
    if let Some(experts) = config.get("num_experts").cloned() {
        config.insert("n_routed_experts".into(), experts);
    }
    for (name, field) in [
        ("expert_shared_count", "n_shared_experts"),
        ("leading_dense_block_count", "first_k_dense_replace"),
        ("expert_group_count", "n_group"),
        ("expert_group_used_count", "topk_group"),
    ] {
        if let Some(value) = header.unsigned(&key(name)) {
            config.insert(field.into(), json!(value));
        }
    }
    if let Some(scale) = header.get(&key("expert_weights_scale")).and_then(decimal) {
        config.insert("routed_scaling_factor".into(), json!(scale));
    }
    let renormalize = header.get(&key("expert_weights_norm")).and_then(MetadataValue::as_bool);
    config.insert("norm_topk_prob".into(), json!(renormalize.unwrap_or(false)));
    // llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` is 2; without a
    // declared gating function llama.cpp routes `deepseek2` by softmax and
    // `glm4moe` by sigmoid.
    let sigmoid = header
        .unsigned(&key("expert_gating_func"))
        .map_or_else(|| architecture == "glm4moe", |gating| gating == 2);
    config.insert(
        "scoring_func".into(),
        json!(if sigmoid {
            "sigmoid"
        } else {
            "softmax"
        }),
    );
    // Softmax routers over several expert groups are DeepSeek-V2's
    // `group_limited_greedy`, which ranks a group by its best expert;
    // llama.cpp ranks every router's groups by their two best experts.
    let grouped = header.unsigned(&key("expert_group_count")).is_some_and(|groups| groups > 1);
    let method = match (sigmoid, grouped) {
        (true, _) => "noaux_tc",
        (false, true) => "group_limited_greedy",
        (false, false) => "greedy",
    };
    config.insert("topk_method".into(), json!(method));
}
