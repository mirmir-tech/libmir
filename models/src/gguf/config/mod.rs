//! The `config.json` a GGUF model's metadata declares.
//!
//! GGUF stores hyperparameters under `<general.architecture>.*` with the same
//! key names for every architecture, so one mapping serves them all. Missing
//! keys stay absent: the decoder layout then reports what it lacks instead of
//! assuming a value.

use serde_json::{Map, Value, json};

mod attention;
mod defaults;
mod experts;
mod gemma4;
mod granite;
mod hybrid;
mod latent;
mod llama3;
mod llama4;
mod long_rope;
mod state_space;

use super::{GgufModel, MetadataValue};
use crate::error::{ModelsError, Result};

/// (metadata key under the architecture prefix, `config.json` key)
const INTEGERS: [(&str, &str); 12] = [
    ("embedding_length", "hidden_size"),
    ("block_count", "num_hidden_layers"),
    ("feed_forward_length", "intermediate_size"),
    ("attention.head_count", "num_attention_heads"),
    ("attention.head_count_kv", "num_key_value_heads"),
    ("attention.key_length", "head_dim"),
    ("context_length", "max_position_embeddings"),
    ("expert_count", "num_experts"),
    ("expert_used_count", "num_experts_per_tok"),
    ("expert_feed_forward_length", "moe_intermediate_size"),
    ("expert_shared_feed_forward_length", "shared_expert_intermediate_size"),
    ("attention.sliding_window", "sliding_window"),
];

const FLOATS: [(&str, &str); 5] = [
    ("attention.layer_norm_rms_epsilon", "rms_norm_eps"),
    ("attention.layer_norm_epsilon", "layer_norm_eps"),
    ("rope.freq_base", "rope_theta"),
    // Gemma 2's score and logit caps (`gemma2.*_softcapping`).
    ("attn_logit_softcapping", "attn_logit_softcapping"),
    ("final_logit_softcapping", "final_logit_softcapping"),
];

/// Whether a GGUF file declaring `general.architecture` `architecture`
/// imports; whether a backend runs it is decided from its header.
#[must_use]
pub fn imports_architecture(architecture: &str) -> bool {
    defaults::imports(architecture)
}

/// Derives `config.json` from the metadata and tensor directory.
pub fn config(model: &GgufModel) -> Result<Value> {
    let header = model.header();
    let architecture = header
        .architecture()
        .ok_or_else(|| ModelsError::InvalidGguf("general.architecture is missing".into()))?;
    let key = |name: &str| format!("{architecture}.{name}");
    let mut config = Map::new();
    // The declared architecture keys import defaults exactly as `model_type`
    // does for SafeTensors checkpoints.
    config.insert("model_type".into(), json!(defaults::model_type(architecture)));
    for (name, field) in INTEGERS {
        // Phi-4 writes `attention.sliding_window = 0` for no window.
        if let Some(value) = header
            .unsigned(&key(name))
            .filter(|&value| value > 0 || field != "sliding_window")
        {
            config.insert(field.into(), json!(value));
        }
    }
    // `block_count` includes multi-token prediction layers after the decoder.
    if let Some(predicted) = header.unsigned(&key("nextn_predict_layers")).filter(|&n| n > 0) {
        let layers = header.unsigned(&key("block_count")).unwrap_or(0);
        let decoder =
            layers.checked_sub(predicted).filter(|&layers| layers > 0).ok_or_else(|| {
                ModelsError::InvalidGguf(format!(
                    "{architecture}: {predicted} prediction layers leave no decoder layers"
                ))
            })?;
        config.insert("num_hidden_layers".into(), json!(decoder));
        config.insert("num_nextn_predict_layers".into(), json!(predicted));
    }
    for (name, field) in FLOATS {
        if let Some(value) = header.get(&key(name)).and_then(decimal) {
            config.insert(field.into(), json!(value));
        }
    }
    granite::granite(header, architecture, &mut config);
    llama4::llama4(header, architecture, &mut config);
    // Cohere multiplies its logits by `logit_scale`.
    if matches!(architecture, "command-r" | "cohere2")
        && let Some(scale) = header.get(&key("logit_scale")).and_then(decimal)
    {
        config.insert("logit_scale".into(), json!(scale));
    }
    long_rope::rope_scaling(model, architecture, &mut config)?;
    // Latent attention declares absorbed key/value widths and a partial
    // rotary key; the latent mapping replaces the per-head checks below.
    let latent = latent::latent(header, architecture, &mut config);
    if !latent && experts::declares_routed_experts(architecture) {
        experts::routed_experts(header, architecture, &mut config);
    }
    // Hybrid DeltaNet decoders rotate part of each head; their mapping
    // declares it as `partial_rotary_factor`.
    let hybrid = hybrid::hybrid(header, architecture, &mut config)?;
    let convolution_bias = model.tensors().any(|tensor| tensor.name.ends_with(".ssm_conv1d.bias"));
    let hybrid =
        state_space::nemotron_h(header, architecture, convolution_bias, &mut config)? || hybrid;
    if !latent {
        head_geometry(header, architecture, hybrid, &mut config)?;
    }
    if let Some(vocabulary) = header.get("tokenizer.ggml.tokens").and_then(MetadataValue::as_array)
    {
        config.insert("vocab_size".into(), json!(vocabulary.len()));
    }
    for (name, field) in [
        ("tokenizer.ggml.bos_token_id", "bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
    ] {
        if let Some(id) = header.unsigned(name) {
            config.insert(field.into(), json!(id));
        }
    }
    if let Some(activation) = defaults::activation(architecture) {
        config.insert("hidden_act".into(), json!(activation));
    }
    defaults::fixed(architecture, &mut config);
    // Gemma 4's `rope_freqs` marks its rotated pairs, not Llama 3 scaling.
    if !gemma4::gemma4(model, architecture, &mut config)? {
        llama3::rope_scaling(model, &mut config)?;
    }
    attention::rope_scaling(header, architecture, &mut config);
    if latent {
        latent::yarn_mscale(header, architecture, &mut config);
        latent::query_temperature(header, architecture, &mut config)?;
    }
    attention::layer_types(header, architecture, &mut config);
    let bias = model.tensors().any(|tensor| {
        tensor.name.ends_with(".attn_q.bias") || tensor.name.ends_with(".attn_qkv.bias")
    });
    config.insert("attention_bias".into(), json!(bias));
    let tied = !model.tensors().any(|tensor| tensor.name == "output.weight");
    config.insert("tie_word_embeddings".into(), json!(tied));
    config.insert("architectures".into(), json!([defaults::model_type(architecture)]));
    // Activations and dense casts run in BF16, as for SafeTensors checkpoints
    // that declare it; each tensor's storage type comes from the directory.
    config.insert("torch_dtype".into(), json!("bfloat16"));
    Ok(Value::Object(config))
}

/// A softmax decoder's head width and rotary part; latent attention
/// declares its own.
fn head_geometry(
    header: &super::GgufHeader,
    architecture: &str,
    hybrid: bool,
    config: &mut Map<String, Value>,
) -> Result<()> {
    let key = |name: &str| format!("{architecture}.{name}");
    let value_length = header.unsigned(&key("attention.value_length"));
    if value_length.is_some() && value_length != header.unsigned(&key("attention.key_length")) {
        return Err(ModelsError::InvalidGguf(format!(
            "{architecture}: attention.value_length differs from key_length; not yet supported"
        )));
    }
    let rotary = header.unsigned(&key("rope.dimension_count"));
    let head_dim = header.unsigned(&key("attention.key_length")).or_else(|| {
        let heads = header.unsigned(&key("attention.head_count")).filter(|&heads| heads > 0)?;
        Some(header.unsigned(&key("embedding_length"))? / heads)
    });
    // A rotary part of each head (GLM-4, Phi-4-mini): its leading
    // `rope.dimension_count` features rotate.
    if !hybrid
        && let (Some(rotary), Some(head_dim)) = (rotary, head_dim)
        && rotary != head_dim
    {
        if rotary == 0 || rotary > head_dim || rotary % 2 != 0 {
            return Err(ModelsError::InvalidGguf(format!(
                "{architecture}: rope.dimension_count {rotary} does not fit head_dim {head_dim}"
            )));
        }
        let factor = f64::from(u32::try_from(rotary)?) / f64::from(u32::try_from(head_dim)?);
        config.insert("partial_rotary_factor".into(), json!(factor));
    }
    // GGUF defines the head width as `embedding_length / head_count` unless
    // `attention.key_length` overrides it (llama.cpp `n_embd_head_k`).
    if let Some(head_dim) = head_dim {
        config.entry("head_dim").or_insert_with(|| json!(head_dim));
    }
    Ok(())
}

/// `generation_config.json` from the special token ids the metadata declares;
/// an end-of-turn token distinct from EOS also ends generation (Gemma's
/// `<end_of_turn>`, as its Hugging Face `generation_config.json` lists).
#[must_use]
pub fn generation_config(model: &GgufModel) -> Value {
    let header = model.header();
    let mut config = Map::new();
    if let Some(id) = header.unsigned("tokenizer.ggml.bos_token_id") {
        config.insert("bos_token_id".into(), json!(id));
    }
    let eos = header.unsigned("tokenizer.ggml.eos_token_id");
    let eot = header.unsigned("tokenizer.ggml.eot_token_id").filter(|&eot| Some(eot) != eos);
    match (eos, eot) {
        (Some(eos), Some(eot)) => config.insert("eos_token_id".into(), json!([eos, eot])),
        (Some(id), None) | (None, Some(id)) => config.insert("eos_token_id".into(), json!(id)),
        (None, None) => None,
    };
    Value::Object(config)
}

/// A float as the converter wrote it: an `f32` by its shortest decimal form,
/// so `1e-6` stays `1e-6` instead of `9.99999997e-7`.
fn decimal(value: &MetadataValue) -> Option<f64> {
    match value {
        MetadataValue::F32(value) => value.to_string().parse().ok(),
        other => other.as_f64(),
    }
}
