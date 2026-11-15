//! Miniature Nemotron-H checkpoint: blocks `mamba, moe, mamba, attention,
//! moe`, in the stacked-expert (MLX) tensor layout.

use std::{fs, path::Path};

use serde_json::{Map, Value, json};

use crate::engine::Result;

pub(super) const HIDDEN: usize = 32;
const INNER: usize = 64;
const CHANNELS: usize = 192;
const PROJECTION: usize = 260;
const EXPERTS: usize = 4;
const EXPERT_INTERMEDIATE: usize = 8;
const SHARED_INTERMEDIATE: usize = 16;

/// `latent` is the routed-expert latent width, `None` for the hidden width.
pub(super) fn write_checkpoint(root: &Path, latent: Option<usize>) -> Result<()> {
    write_config(root, latent)?;
    write_weights(&root.join("model.safetensors"), latent)
}

fn write_config(root: &Path, latent: Option<usize>) -> Result<()> {
    let config = json!({
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hidden_size": HIDDEN,
        "intermediate_size": EXPERT_INTERMEDIATE,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "layer_norm_epsilon": 1e-5,
        "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 32,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "expand": 2,
        "mamba_ssm_cache_dtype": "float32",
        "use_conv_bias": true,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "moe_latent_size": latent,
        "moe_intermediate_size": EXPERT_INTERMEDIATE,
        "moe_shared_expert_intermediate_size": SHARED_INTERMEDIATE,
        "n_shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "mlp_hidden_act": "relu2",
        "tie_word_embeddings": false
    });
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    Ok(())
}

fn specs(latent: Option<usize>) -> Vec<(String, Vec<usize>)> {
    let (hidden, intermediate, shared) = (HIDDEN, EXPERT_INTERMEDIATE, SHARED_INTERMEDIATE);
    let mut specs: Vec<(String, Vec<usize>)> = vec![
        ("backbone.embeddings.weight".into(), vec![64, hidden]),
        ("backbone.norm_f.weight".into(), vec![hidden]),
        ("lm_head.weight".into(), vec![64, hidden]),
    ];
    for block in [0_usize, 2] {
        let root = format!("backbone.layers.{block}");
        specs.push((format!("{root}.norm.weight"), vec![hidden]));
        for name in ["A_log", "D", "dt_bias"] {
            specs.push((format!("{root}.mixer.{name}"), vec![4]));
        }
        specs.push((format!("{root}.mixer.conv1d.weight"), vec![CHANNELS, 4, 1]));
        specs.push((format!("{root}.mixer.conv1d.bias"), vec![CHANNELS]));
        specs.push((format!("{root}.mixer.in_proj.weight"), vec![PROJECTION, hidden]));
        specs.push((format!("{root}.mixer.norm.weight"), vec![INNER]));
        specs.push((format!("{root}.mixer.out_proj.weight"), vec![hidden, INNER]));
    }
    specs.push(("backbone.layers.3.norm.weight".into(), vec![hidden]));
    specs.push(("backbone.layers.3.mixer.q_proj.weight".into(), vec![hidden, hidden]));
    specs.push(("backbone.layers.3.mixer.k_proj.weight".into(), vec![16, hidden]));
    specs.push(("backbone.layers.3.mixer.v_proj.weight".into(), vec![16, hidden]));
    specs.push(("backbone.layers.3.mixer.o_proj.weight".into(), vec![hidden, hidden]));
    for block in [1_usize, 4] {
        let root = format!("backbone.layers.{block}");
        let width = latent.unwrap_or(hidden);
        specs.push((format!("{root}.norm.weight"), vec![hidden]));
        specs.push((format!("{root}.mixer.gate.weight"), vec![EXPERTS, hidden]));
        specs.push((format!("{root}.mixer.gate.e_score_correction_bias"), vec![EXPERTS]));
        specs.push((
            format!("{root}.mixer.switch_mlp.fc1.weight"),
            vec![EXPERTS, intermediate, width],
        ));
        specs.push((
            format!("{root}.mixer.switch_mlp.fc2.weight"),
            vec![EXPERTS, width, intermediate],
        ));
        specs.push((format!("{root}.mixer.shared_experts.up_proj.weight"), vec![shared, hidden]));
        specs.push((format!("{root}.mixer.shared_experts.down_proj.weight"), vec![hidden, shared]));
        if let Some(latent) = latent {
            specs.push((format!("{root}.mixer.fc1_latent_proj.weight"), vec![latent, hidden]));
            specs.push((format!("{root}.mixer.fc2_latent_proj.weight"), vec![hidden, latent]));
        }
    }
    specs
}

fn write_weights(path: &Path, latent: Option<usize>) -> Result<()> {
    let mut header = Map::new();
    let mut offset = 0_usize;
    let mut payload = Vec::new();
    let mut seed = 0x9E37_79B9_u32;
    for (name, shape) in specs(latent) {
        let elements = shape.iter().product::<usize>();
        let bytes = elements * 4;
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        for _ in 0..elements {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let unit = f32::from(u16::try_from(seed % 2048)?) / 1024.0 - 1.0;
            payload.extend_from_slice(&(unit * 0.25).to_le_bytes());
        }
        offset += bytes;
    }
    let mut encoded = serde_json::to_string(&Value::Object(header))?;
    while !encoded.len().is_multiple_of(8) {
        encoded.push(' ');
    }
    let mut bytes = u64::try_from(encoded.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(encoded.as_bytes());
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes)?;
    Ok(())
}
