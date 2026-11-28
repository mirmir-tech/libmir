//! GGUF tensor names mapped to the `SafeTensors` names the binding layer
//! reads. GGUF names its tensors by role (`gguf-py` `TENSOR_NAMES`), the same
//! for every architecture, so the mapping is one table of roles, not of models.

mod prediction;
mod state_space;

use std::collections::BTreeSet;

use super::{GgufHeader, vision::Projector};
use crate::{
    error::Result,
    weights::{DenseDType, TensorInfo},
};

/// Whole-model tensors; Phi-3.5-MoE's final `LayerNorm` and output carry
/// biases.
const GLOBAL: [(&str, &str); 8] = [
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output_norm.bias", "model.norm.bias"),
    ("output.weight", "lm_head.weight"),
    ("output.bias", "lm_head.bias"),
    // Gemma 4's E models: the per-layer token embedding and the projection
    // of the token embeddings into every layer's input.
    ("per_layer_token_embd.weight", "model.embed_tokens_per_layer.weight"),
    ("per_layer_model_proj.weight", "model.per_layer_model_projection.weight"),
    ("per_layer_proj_norm.weight", "model.per_layer_projection_norm.weight"),
];

/// Per-block tensors: `blk.N.<role>` → `model.layers.N.<name>`.
const BLOCK: [(&str, &str); 45] = [
    ("attn_norm", "input_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_gate_inp", "mlp.gate"),
    ("post_attention_norm", "post_attention_layernorm"),
    (POST_FEED_FORWARD_NORMS[0], "post_feedforward_layernorm"),
    // GLM-4 names the same sandwich norms after their sub-block.
    ("attn_post_norm", "post_attention_layernorm"),
    (POST_FEED_FORWARD_NORMS[1], "post_feedforward_layernorm"),
    // Gemma 4's parallel dense and expert feed-forwards.
    ("pre_ffw_norm_2", "pre_feedforward_layernorm_2"),
    ("post_ffw_norm_1", "post_feedforward_layernorm_1"),
    ("post_ffw_norm_2", "post_feedforward_layernorm_2"),
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
    ("ffn_gate_up_exps", "mlp.experts.gate_up_proj"),
    // Gemma 4's E models: each layer's per-layer input gate.
    ("inp_gate", "per_layer_input_gate"),
    ("proj", "per_layer_projection"),
    ("post_norm", "post_per_layer_input_norm"),
    // Multi-head latent attention. Recent converters split
    // `attn_kv_b` per head into the absorbed key and value up-projections.
    ("attn_q_a", "self_attn.q_a_proj"),
    ("attn_q_a_norm", "self_attn.q_a_layernorm"),
    ("attn_q_b", "self_attn.q_b_proj"),
    ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
    ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
    ("attn_kv_b", "self_attn.kv_b_proj"),
    ("attn_k_b", "self_attn.k_b_proj"),
    ("attn_v_b", "self_attn.v_b_proj"),
    ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
    ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
    ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
    ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
    // Gated DeltaNet (Qwen3.5 `qwen35`): llama.cpp names its projections by
    // what they compute; `attn_gate` is the output gate `z`.
    ("attn_qkv", "linear_attn.in_proj_qkv"),
    ("attn_gate", "linear_attn.in_proj_z"),
    ("ssm_alpha", "linear_attn.in_proj_a"),
    ("ssm_beta", "linear_attn.in_proj_b"),
    ("ssm_conv1d", "linear_attn.conv1d"),
    ("ssm_norm", "linear_attn.norm"),
    ("ssm_out", "linear_attn.out_proj"),
];

/// The roles of the norm after a block's feed-forward: Gemma's, then GLM-4's.
const POST_FEED_FORWARD_NORMS: [&str; 2] = ["post_ffw_norm", "ffn_post_norm"];

/// In a block with both output norms (Gemma's sandwich), `ffn_norm` is the
/// norm before the feed-forward rather than after attention.
pub const SANDWICH_FEED_FORWARD_NORM: (&str, &str) = ("ffn_norm", "pre_feedforward_layernorm");

/// Roles whose `SafeTensors` name carries no `.weight` suffix.
const BARE: [(&str, &str); 5] = [
    ("attn_sinks.weight", "self_attn.sinks"),
    ("exp_probs_b.bias", "mlp.gate.e_score_correction_bias"),
    // llama.cpp stores `-exp(A_log)` here (`LinearAttentionLayout::LlamaCpp`).
    ("ssm_a", "linear_attn.A_log"),
    ("ssm_dt.bias", "linear_attn.dt_bias"),
    ("layer_output_scale.weight", "layer_scalar"),
];

/// The `SafeTensors` name of a GGUF tensor; `None` for a role this table does
/// not map yet, which admission reports instead of guessing.
#[must_use]
pub fn safetensors_name(gguf: &str) -> Option<String> {
    if let Some((_, name)) = GLOBAL.iter().find(|(from, _)| *from == gguf) {
        return Some((*name).to_owned());
    }
    let rest = gguf.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    if layer.is_empty() || !layer.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if let Some((_, name)) = BARE.iter().find(|(from, _)| *from == rest) {
        return Some(format!("model.layers.{layer}.{name}"));
    }
    let (role, suffix) = rest.rsplit_once('.')?;
    let (_, name) = BLOCK.iter().find(|(from, _)| *from == role)?;
    Some(format!("model.layers.{layer}.{name}.{suffix}"))
}

/// Renames a GGUF catalog to `SafeTensors` names in place: an `mmproj` file
/// by its projector's names, a text model by the role tables and the layout
/// fixes below.
pub(super) fn map(header: &GgufHeader, tensors: &mut [TensorInfo]) -> Result<()> {
    if header.architecture() == Some(super::config::NEMOTRON_H) {
        state_space::map(tensors);
        return Ok(());
    }
    if let Some(projector) = Projector::of(header)? {
        for tensor in tensors {
            if let Some(name) = projector.tensor_name(&tensor.name) {
                tensor.name = name;
            }
        }
        return Ok(());
    }
    // The blocks with a norm after their feed-forward, indexed over the whole
    // model before any tensor is renamed: a shard may hold a block's
    // `ffn_norm` and another its companion.
    let sandwiches = tensors
        .iter()
        .filter_map(|tensor| {
            let (block, role) = tensor.name.strip_suffix(".weight")?.rsplit_once('.')?;
            POST_FEED_FORWARD_NORMS.contains(&role).then(|| block.to_owned())
        })
        .collect::<BTreeSet<_>>();
    let sandwich = |name: &str| {
        let (block, role) = name.strip_suffix(".weight")?.rsplit_once('.')?;
        if role != SANDWICH_FEED_FORWARD_NORM.0 {
            return None;
        }
        let layer = block.strip_prefix("blk.")?;
        sandwiches
            .contains(block)
            .then(|| format!("model.layers.{layer}.{}.weight", SANDWICH_FEED_FORWARD_NORM.1))
    };
    for tensor in tensors.iter_mut() {
        if let Some(name) = sandwich(&tensor.name).or_else(|| safetensors_name(&tensor.name)) {
            tensor.name = name;
        }
        // `ffn_gate_inp_shexp` drops the unit output axis of the shared
        // expert's gate, a `[1, hidden]` projection in the checkpoint.
        if tensor.name.ends_with(".mlp.shared_expert_gate.weight")
            && let [hidden] = *tensor.shape.as_slice()
        {
            tensor.shape = vec![1, hidden];
        }
    }
    // Gemma 4's `ffn_gate_inp.scale` is the router's input norm scale, and
    // `ffn_down_exps.scale` its `per_expert_scale`: a factor on each expert's
    // output is one on its routing weight, which the runtime applies once.
    if header.architecture() == Some("gemma4") {
        for tensor in tensors.iter_mut() {
            if let Some(stem) = tensor.name.strip_suffix(".mlp.gate.scale") {
                tensor.name = format!("{stem}.router.scale");
            } else if let Some(stem) = tensor.name.strip_suffix(".mlp.experts.down_proj.scale") {
                tensor.name = format!("{stem}.router.per_expert_scale");
            }
        }
    }
    prediction::map(header, tensors);
    // llama.cpp's FP8 `.scale` multiplies the products, as a checkpoint's
    // `weight_scale` does.
    let fp8 = tensors
        .iter()
        .filter(|tensor| tensor.dtype == DenseDType::F8E4M3)
        .filter_map(|tensor| tensor.name.strip_suffix(".weight").map(str::to_owned))
        .collect::<BTreeSet<_>>();
    for tensor in tensors {
        if let Some(stem) = tensor.name.strip_suffix(".scale")
            && fp8.contains(stem)
        {
            tensor.name = format!("{stem}.weight_scale");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::safetensors_name;

    #[test]
    fn maps_roles_not_models() {
        assert_eq!(
            safetensors_name("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            safetensors_name("blk.12.attn_q_norm.weight").as_deref(),
            Some("model.layers.12.self_attn.q_norm.weight")
        );
        assert_eq!(
            safetensors_name("blk.0.ffn_down.weight").as_deref(),
            Some("model.layers.0.mlp.down_proj.weight")
        );
        assert_eq!(
            safetensors_name("blk.4.ffn_up_exps.bias").as_deref(),
            Some("model.layers.4.mlp.experts.up_proj.bias")
        );
        assert_eq!(
            safetensors_name("blk.4.attn_sinks.weight").as_deref(),
            Some("model.layers.4.self_attn.sinks")
        );
        assert_eq!(safetensors_name("blk.x.attn_q.weight"), None);
        assert_eq!(safetensors_name("blk.3.unknown_role.weight"), None);
        assert_eq!(safetensors_name("rope_freqs.weight"), None);
        // Phi-3.5-MoE's biased final `LayerNorm` and output projection.
        assert_eq!(safetensors_name("output_norm.bias").as_deref(), Some("model.norm.bias"));
        assert_eq!(safetensors_name("output.bias").as_deref(), Some("lm_head.bias"));
    }
}
