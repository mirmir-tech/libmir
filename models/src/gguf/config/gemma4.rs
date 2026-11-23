//! Gemma 4 text decoders (`gemma4`) as the Hugging Face `text_config`
//! declares them: sliding layers with `head_dim` and full layers with
//! `global_head_dim` and fewer K/V heads, proportional `RoPE` on full layers,
//! K/V sharing where full layers have no value projection, and the final
//! logit softcap. llama.cpp's converter (`conversion/gemma.py`,
//! `Gemma4Model`) keeps the norms as they are and marks the rotated pairs of
//! full layers in `rope_freqs` (1 for rotated, 1e30 for the others).

use serde_json::{Map, Value, json};

use super::{
    super::{GgufModel, MetadataValue},
    decimal, llama3,
};
use crate::error::{ModelsError, Result};

/// Whether `architecture` is Gemma 4; fills its keys.
pub(super) fn gemma4(
    model: &GgufModel,
    architecture: &str,
    config: &mut Map<String, Value>,
) -> Result<bool> {
    if architecture != "gemma4" {
        return Ok(false);
    }
    let header = model.header();
    let key = |name: &str| format!("gemma4.{name}");
    let missing = |name: &str| ModelsError::InvalidGguf(format!("gemma4: {name} is missing"));
    let unsigned = |name: &str| header.unsigned(&key(name)).ok_or_else(|| missing(name));
    let sliding = header
        .get(&key("attention.sliding_window_pattern"))
        .and_then(MetadataValue::as_array)
        .map(|layers| {
            layers
                .iter()
                .map(|layer| *layer == MetadataValue::Bool(true))
                .collect::<Vec<_>>()
        })
        .ok_or_else(|| missing("attention.sliding_window_pattern"))?;
    let types = sliding
        .iter()
        .map(|&sliding| {
            if sliding {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect::<Vec<_>>();
    config.insert("layer_types".into(), json!(types));
    let heads = header
        .get(&key("attention.head_count_kv"))
        .and_then(MetadataValue::as_array)
        .map(|heads| heads.iter().filter_map(MetadataValue::as_u64).collect::<Vec<_>>())
        .filter(|heads| heads.len() == sliding.len())
        .ok_or_else(|| missing("attention.head_count_kv per layer"))?;
    let first = |want: bool| {
        sliding
            .iter()
            .zip(&heads)
            .find(|(sliding, _)| **sliding == want)
            .map(|(_, &heads)| heads)
    };
    config.insert(
        "num_key_value_heads".into(),
        json!(first(true).ok_or_else(|| missing("a sliding layer"))?),
    );
    if let Some(global) = first(false) {
        config.insert("num_global_key_value_heads".into(), json!(global));
    }
    let (global_dim, sliding_dim) =
        (unsigned("attention.key_length")?, unsigned("attention.key_length_swa")?);
    config.insert("head_dim".into(), json!(sliding_dim));
    config.insert("global_head_dim".into(), json!(global_dim));
    rope_parameters(model, global_dim, config)?;
    // Full layers without a value projection attend with values equal to
    // their keys.
    let shares = sliding.iter().enumerate().any(|(layer, &sliding)| {
        !sliding
            && !model
                .tensors()
                .any(|tensor| tensor.name == format!("blk.{layer}.attn_v.weight"))
    });
    config.insert("attention_k_eq_v".into(), json!(shares));
    if let Some(softcap) = header.get(&key("final_logit_softcapping")).and_then(decimal) {
        config.insert("final_logit_softcapping".into(), json!(softcap));
    }
    config.insert("hidden_activation".into(), json!("gelu_pytorch_tanh"));
    experts(model, config);
    for (name, field) in [
        ("attention.shared_kv_layers", "num_kv_shared_layers"),
        ("embedding_length_per_layer_input", "hidden_size_per_layer_input"),
    ] {
        if let Some(value) = header.unsigned(&key(name)) {
            config.insert(field.into(), json!(value));
        }
    }
    Ok(true)
}

/// `rope_parameters` per layer type: proportional on full layers, over the
/// share of pairs `rope_freqs` marks as rotated.
fn rope_parameters(
    model: &GgufModel,
    global_dim: u64,
    config: &mut Map<String, Value>,
) -> Result<()> {
    let header = model.header();
    let theta = |name: &str| header.get(&format!("gemma4.{name}")).and_then(decimal);
    // The rotated share of a full layer's head: its `rope_freqs` pairs of 1.
    let rotated = model
        .shards
        .iter()
        .find_map(|(path, header)| {
            let tensor = header.tensors.iter().find(|tensor| tensor.name == llama3::TENSOR)?;
            Some(llama3::read(path, header.data_start, tensor))
        })
        .transpose()?
        .map(|factors| factors.iter().filter(|&&factor| (factor - 1.0).abs() < 1e-6).count());
    let full_factor = rotated.map_or(1.0, |pairs| {
        f64::from(u32::try_from(pairs * 2).unwrap_or(0))
            / f64::from(u32::try_from(global_dim).unwrap_or(1))
    });
    let mut full = json!({"rope_type": "proportional", "partial_rotary_factor": full_factor});
    if let Some(theta) = theta("rope.freq_base") {
        full["rope_theta"] = json!(theta);
    }
    let mut local = json!({"rope_type": "default"});
    if let Some(theta) = theta("rope.freq_base_swa") {
        local["rope_theta"] = json!(theta);
    }
    config.insert(
        "rope_parameters".into(),
        json!({"full_attention": full, "sliding_attention": local}),
    );
    Ok(())
}

/// The routed experts of the mixture-of-experts variant (`26B-A4B`).
fn experts(model: &GgufModel, config: &mut Map<String, Value>) {
    let header = model.header();
    let key = |name: &str| format!("gemma4.{name}");
    if let Some(experts) = header.unsigned(&key("expert_count")).filter(|&experts| experts > 0) {
        config.insert("enable_moe_block".into(), json!(true));
        config.insert("num_experts".into(), json!(experts));
        if let Some(top_k) = header.unsigned(&key("expert_used_count")) {
            config.insert("top_k_experts".into(), json!(top_k));
        }
    }
}
