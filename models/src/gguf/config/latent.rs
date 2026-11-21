//! Multi-head latent attention and its shared-routed experts under the
//! Hugging Face latent-attention config names (`kv_lora_rank`, …).
//!
//! llama.cpp declares the absorbed geometry: `attention.key_length` and
//! `value_length` are the cached latent row (`kv_lora_rank + rope`) and the
//! latent, while `key_length_mla` and `value_length_mla` are each head's
//! query/key and value widths. Older conversions stored the head widths in
//! `key_length` and `value_length` and an unsplit `attn_kv_b`.

use serde_json::{Map, Value, json};

use super::{
    super::{GgufHeader, MetadataValue},
    decimal,
};
use crate::error::{ModelsError, Result};

/// Inserts the latent geometry and expert routing when the metadata
/// declares a key/value latent; returns whether it did.
pub(super) fn latent(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) -> bool {
    let key = |name: &str| format!("{architecture}.{name}");
    let Some(rank) = header.unsigned(&key("attention.kv_lora_rank")) else {
        return false;
    };
    let rope = header.unsigned(&key("rope.dimension_count"));
    let (query_key, value) = header.unsigned(&key("attention.key_length_mla")).map_or_else(
        || {
            (
                header.unsigned(&key("attention.key_length")),
                header.unsigned(&key("attention.value_length")),
            )
        },
        |query_key| (Some(query_key), header.unsigned(&key("attention.value_length_mla"))),
    );
    config.insert("kv_lora_rank".into(), json!(rank));
    if let Some(query_rank) =
        header.unsigned(&key("attention.q_lora_rank")).filter(|&rank| rank > 0)
    {
        config.insert("q_lora_rank".into(), json!(query_rank));
    }
    if let (Some(rope), Some(query_key)) = (rope, query_key) {
        config.insert("qk_rope_head_dim".into(), json!(rope));
        config.insert("qk_nope_head_dim".into(), json!(query_key.saturating_sub(rope)));
    }
    if let Some(value) = value {
        config.insert("v_head_dim".into(), json!(value));
    }
    // `key_length` is the absorbed row, not a head width, and the absorbed
    // graph has one key/value head; the Hugging Face config has neither.
    drop(config.remove("head_dim"));
    if let Some(heads) = config.get("num_attention_heads").cloned() {
        config.insert("num_key_value_heads".into(), heads);
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
    // declared gating function llama.cpp routes by softmax.
    let sigmoid = header.unsigned(&key("expert_gating_func")) == Some(2);
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
    true
}

/// The converter writes `0.1 · mscale_all_dim` as `yarn_log_multiplier`
/// (llama.cpp `TAG_DEEPSEEK2_YARN_LOG_MUL_FIX`) and records no separate
/// `mscale`; the converted checkpoints set both equal, so the rotary
/// attention factor is one.
pub(super) fn yarn_mscale(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) {
    let name = format!("{architecture}.rope.scaling.yarn_log_multiplier");
    let (Some(multiplier), Some(Value::Object(scaling))) =
        (header.get(&name).and_then(decimal), config.get_mut("rope_scaling"))
    else {
        return;
    };
    let mscale = multiplier / 0.1;
    scaling.insert("mscale_all_dim".into(), json!(mscale));
    scaling.insert("mscale".into(), json!(mscale));
}

/// llama.cpp's attention temperature (`attention.temperature_scale` over
/// `temperature_length` positions) is Mistral 4's `llama_4_scaling_beta`
/// over the `YaRN` window; any other window has no config form.
pub(super) fn query_temperature(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) -> Result<()> {
    let key = |name: &str| format!("{architecture}.{name}");
    let Some(beta) = header.get(&key("attention.temperature_scale")).and_then(decimal) else {
        return Ok(());
    };
    let length = header.unsigned(&key("attention.temperature_length"));
    let Some(Value::Object(scaling)) = config.get_mut("rope_scaling") else {
        return Err(unsupported(architecture));
    };
    if length.is_none()
        || length != scaling.get("original_max_position_embeddings").and_then(Value::as_u64)
    {
        return Err(unsupported(architecture));
    }
    scaling.insert("llama_4_scaling_beta".into(), json!(beta));
    Ok(())
}

fn unsupported(architecture: &str) -> ModelsError {
    ModelsError::InvalidGguf(format!(
        "{architecture}: attention temperature outside the YaRN window is not yet supported"
    ))
}
