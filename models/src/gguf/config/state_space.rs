//! Nemotron-H (`nemotron_h`): Mamba-2, attention and MLP blocks, one per GGUF
//! block. llama.cpp marks an attention block by a nonzero
//! `attention.head_count_kv` and an MLP block by a nonzero
//! `feed_forward_length` (`llama_model_nemotron_h`, v1.0.0); every other
//! block is Mamba-2. The Hugging Face config names them in
//! `layers_block_type`.

use serde_json::{Map, Value, json};

use super::super::{GgufHeader, MetadataValue};
use crate::{
    error::{ModelsError, Result},
    layout::{STATE_SPACE_LAYOUT, STATE_SPACE_LAYOUT_LLAMA_CPP},
};

/// Whether `architecture` is Nemotron-H; fills its keys.
pub(super) fn nemotron_h(
    header: &GgufHeader,
    architecture: &str,
    has_convolution_bias: bool,
    config: &mut Map<String, Value>,
) -> Result<bool> {
    if architecture != "nemotron_h" {
        return Ok(false);
    }
    let key = |name: &str| format!("{architecture}.{name}");
    let invalid = |detail: &str| ModelsError::InvalidGguf(format!("{architecture}: {detail}"));
    let per_layer = |name: &str| {
        header
            .get(&key(name))
            .and_then(MetadataValue::as_array)
            .map(|values| {
                values.iter().map(|value| value.as_u64().unwrap_or(0)).collect::<Vec<_>>()
            })
            .ok_or_else(|| invalid(&format!("{name} must be a per-layer array")))
    };
    let key_value_heads = per_layer("attention.head_count_kv")?;
    let feed_forward = per_layer("feed_forward_length")?;
    if key_value_heads.len() != feed_forward.len() {
        return Err(invalid("per-layer arrays differ in length"));
    }
    let blocks = key_value_heads
        .iter()
        .zip(&feed_forward)
        .map(|(&heads, &width)| match (heads, width) {
            (0, 0) => Ok("mamba"),
            (_, 0) => Ok("attention"),
            (0, _) => Ok("mlp"),
            _ => Err(invalid("a block has both attention and an MLP")),
        })
        .collect::<Result<Vec<_>>>()?;
    config.insert("layers_block_type".into(), json!(blocks));
    config.insert("num_key_value_heads".into(), json!(key_value_heads.iter().max()));
    config.insert("intermediate_size".into(), json!(feed_forward.iter().max()));
    let unsigned = |name: &str| {
        header
            .unsigned(&key(name))
            .ok_or_else(|| invalid(&format!("{name} is missing")))
    };
    let heads = unsigned("ssm.time_step_rank")?;
    let inner = unsigned("ssm.inner_size")?;
    if heads == 0 || inner % heads != 0 {
        return Err(invalid("ssm.inner_size does not split into ssm.time_step_rank heads"));
    }
    for (field, value) in [
        ("mamba_num_heads", heads),
        ("mamba_head_dim", inner / heads),
        ("ssm_state_size", unsigned("ssm.state_size")?),
        ("n_groups", unsigned("ssm.group_count")?),
        ("conv_kernel", unsigned("ssm.conv_kernel")?),
        // The SSD chunk is a kernel choice GGUF does not record; NVIDIA's
        // Nemotron-H configs use 128.
        ("chunk_size", 128),
    ] {
        config.insert(field.into(), json!(value));
    }
    config.insert("use_conv_bias".into(), json!(has_convolution_bias));
    config.insert("mamba_proj_bias".into(), json!(false));
    config.insert("mamba_hidden_act".into(), json!("silu"));
    config.insert("mlp_hidden_act".into(), json!("relu2"));
    config.insert(STATE_SPACE_LAYOUT.into(), json!(STATE_SPACE_LAYOUT_LLAMA_CPP));
    Ok(true)
}
