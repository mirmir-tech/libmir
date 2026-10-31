//! Miniature state-space hybrid checkpoint with the Nemotron-H tensor schema:
//! blocks `mamba, moe, mamba, attention, moe` form three semantic layers.

use serde_json::{Value, json};

use super::super::tensor;
use crate::weights::{TensorCatalog, TensorInfo};

pub(super) const HIDDEN: usize = 32;
pub(super) const EXPERTS: usize = 4;
const LATENT: usize = 16;
const EXPERT_WIDTH: usize = 32;
const INNER: usize = 64;
const CONVOLUTION: usize = 80;
const INPUT_PROJECTION: usize = 148;

pub(super) fn config() -> Value {
    json!({
        "llm_config": {
            "hidden_size": HIDDEN,
            "intermediate_size": EXPERT_WIDTH,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "layer_norm_epsilon": 1e-5,
            "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "ssm_state_size": 4,
            "n_groups": 2,
            "conv_kernel": 4,
            "chunk_size": 8,
            "expand": 2,
            "time_step_limit": [0.0, { "__float__": "Infinity" }],
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "moe_latent_size": LATENT,
            "moe_intermediate_size": EXPERT_WIDTH,
            "moe_shared_expert_intermediate_size": EXPERT_WIDTH,
            "n_shared_experts": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "mlp_hidden_act": "relu2"
        }
    })
}

pub(super) fn catalog() -> TensorCatalog {
    let mut tensors = Vec::new();
    dense(&mut tensors, "language_model.backbone.embeddings.weight", &[64, HIDDEN]);
    dense(&mut tensors, "language_model.backbone.norm_f.weight", &[HIDDEN]);
    float8(&mut tensors, "language_model.lm_head", &[64, HIDDEN], &[]);
    mamba(&mut tensors, 0);
    moe(&mut tensors, 1);
    mamba(&mut tensors, 2);
    attention(&mut tensors, 3);
    moe(&mut tensors, 4);
    dense(
        &mut tensors,
        "language_model.mtp.layers.0.eh_proj.weight",
        &[HIDDEN, 2 * HIDDEN],
    );
    dense(&mut tensors, "vision_model.radio_model.patch.weight", &[8, 8]);
    sorted(tensors)
}

pub(super) fn sorted(mut tensors: Vec<TensorInfo>) -> TensorCatalog {
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    TensorCatalog { tensors }
}

pub(super) fn block(index: usize) -> String {
    format!("language_model.backbone.layers.{index}")
}

fn mamba(tensors: &mut Vec<TensorInfo>, index: usize) {
    let root = format!("{}.mixer", block(index));
    dense(tensors, &format!("{}.norm.weight", block(index)), &[HIDDEN]);
    for name in ["A_log", "D", "dt_bias"] {
        tensors.push(tensor(&format!("{root}.{name}"), "F32", vec![4]));
    }
    dense(tensors, &format!("{root}.conv1d.weight"), &[CONVOLUTION, 1, 4]);
    dense(tensors, &format!("{root}.conv1d.bias"), &[CONVOLUTION]);
    dense(tensors, &format!("{root}.norm.weight"), &[INNER]);
    // Rank-0 and rank-1 tensor scales both describe one scalar.
    float8(tensors, &format!("{root}.in_proj"), &[INPUT_PROJECTION, HIDDEN], &[]);
    float8(tensors, &format!("{root}.out_proj"), &[HIDDEN, INNER], &[1]);
}

fn attention(tensors: &mut Vec<TensorInfo>, index: usize) {
    let root = format!("{}.mixer", block(index));
    dense(tensors, &format!("{}.norm.weight", block(index)), &[HIDDEN]);
    float8(tensors, &format!("{root}.q_proj"), &[HIDDEN, HIDDEN], &[1]);
    float8(tensors, &format!("{root}.k_proj"), &[16, HIDDEN], &[]);
    float8(tensors, &format!("{root}.v_proj"), &[16, HIDDEN], &[]);
    float8(tensors, &format!("{root}.o_proj"), &[HIDDEN, HIDDEN], &[1]);
}

fn moe(tensors: &mut Vec<TensorInfo>, index: usize) {
    let root = format!("{}.mixer", block(index));
    dense(tensors, &format!("{}.norm.weight", block(index)), &[HIDDEN]);
    dense(tensors, &format!("{root}.gate.weight"), &[EXPERTS, HIDDEN]);
    tensors.push(tensor(&format!("{root}.gate.e_score_correction_bias"), "F32", vec![EXPERTS]));
    dense(tensors, &format!("{root}.fc1_latent_proj.weight"), &[LATENT, HIDDEN]);
    dense(tensors, &format!("{root}.fc2_latent_proj.weight"), &[HIDDEN, LATENT]);
    for expert in 0..EXPERTS {
        nvfp4(tensors, &format!("{root}.experts.{expert}.up_proj"), EXPERT_WIDTH, LATENT);
        nvfp4(tensors, &format!("{root}.experts.{expert}.down_proj"), LATENT, EXPERT_WIDTH);
    }
    nvfp4(tensors, &format!("{root}.shared_experts.up_proj"), EXPERT_WIDTH, HIDDEN);
    nvfp4(tensors, &format!("{root}.shared_experts.down_proj"), HIDDEN, EXPERT_WIDTH);
}

pub(super) fn dense(tensors: &mut Vec<TensorInfo>, name: &str, shape: &[usize]) {
    tensors.push(tensor(name, "BF16", shape.to_vec()));
}

fn float8(tensors: &mut Vec<TensorInfo>, prefix: &str, shape: &[usize], scale: &[usize]) {
    tensors.push(tensor(&format!("{prefix}.weight"), "F8_E4M3", shape.to_vec()));
    tensors.push(tensor(&format!("{prefix}.weight_scale"), "F32", scale.to_vec()));
    tensors.push(tensor(&format!("{prefix}.input_scale"), "F32", scale.to_vec()));
}

fn nvfp4(tensors: &mut Vec<TensorInfo>, prefix: &str, output: usize, input: usize) {
    tensors.push(tensor(&format!("{prefix}.weight"), "U8", vec![output, input / 2]));
    tensors.push(tensor(&format!("{prefix}.weight_scale"), "F8_E4M3", vec![output, input / 16]));
    tensors.push(tensor(&format!("{prefix}.weight_scale_2"), "F32", Vec::new()));
    tensors.push(tensor(&format!("{prefix}.input_scale"), "F32", Vec::new()));
}
