use std::{fs, path::Path};

use serde_json::{Map, Value, json};

use crate::engine::Result;

pub fn write_config(root: &Path) -> Result<()> {
    let config = json!({
        "architectures": ["GptOssForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_act": "silu",
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "layer_types": ["full_attention"],
        "rope_theta": 150_000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 32
        },
        "tie_word_embeddings": false
    });
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    Ok(())
}

pub fn write_weights(path: &Path, fused: bool) -> Result<()> {
    let layer = "model.layers.0";
    let mut specs = vec![
        ("model.embed_tokens.weight".into(), vec![64, 32]),
        ("model.norm.weight".into(), vec![32]),
        ("lm_head.weight".into(), vec![64, 32]),
        (format!("{layer}.input_layernorm.weight"), vec![32]),
        (format!("{layer}.self_attn.q_proj.weight"), vec![32, 32]),
        (format!("{layer}.self_attn.k_proj.weight"), vec![16, 32]),
        (format!("{layer}.self_attn.v_proj.weight"), vec![16, 32]),
        (format!("{layer}.self_attn.o_proj.weight"), vec![32, 32]),
        (format!("{layer}.self_attn.sinks"), vec![4]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![32]),
        (format!("{layer}.mlp.router.weight"), vec![2, 32]),
    ];
    if fused {
        specs.extend([
            (format!("{layer}.mlp.experts.gate_up_proj"), vec![2, 32, 64]),
            (format!("{layer}.mlp.experts.gate_up_proj_bias"), vec![2, 64]),
            (format!("{layer}.mlp.experts.down_proj"), vec![2, 32, 32]),
            (format!("{layer}.mlp.experts.down_proj_bias"), vec![2, 32]),
        ]);
    } else {
        specs.extend([
            (format!("{layer}.mlp.experts.gate_proj.weight"), vec![2, 32, 32]),
            (format!("{layer}.mlp.experts.up_proj.weight"), vec![2, 32, 32]),
            (format!("{layer}.mlp.experts.down_proj.weight"), vec![2, 32, 32]),
        ]);
    }
    for (name, shape) in [
        (format!("{layer}.self_attn.q_proj.bias"), vec![32]),
        (format!("{layer}.self_attn.k_proj.bias"), vec![16]),
        (format!("{layer}.self_attn.v_proj.bias"), vec![16]),
        (format!("{layer}.self_attn.o_proj.bias"), vec![32]),
    ] {
        specs.push((name, shape));
    }
    let mut header = Map::new();
    let mut offset = 0_usize;
    let mut payload = Vec::new();
    for (name, shape) in &specs {
        let elements = shape.iter().product::<usize>();
        let bytes = elements.checked_mul(4).ok_or(crate::engine::Error::ShapeOverflow)?;
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        for index in 0..elements {
            let value = if name.contains("norm.weight") {
                1.0
            } else {
                (f32::from(u16::try_from((index * 13 + offset) % 251)?) / 251.0 - 0.5) * 0.2
            };
            payload.extend_from_slice(&value.to_le_bytes());
        }
        offset += bytes;
    }
    write_safetensors(path, header, &payload)
}

fn write_safetensors(path: &Path, header: Map<String, Value>, payload: &[u8]) -> Result<()> {
    let mut header = serde_json::to_string(&Value::Object(header))?;
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes)?;
    Ok(())
}
