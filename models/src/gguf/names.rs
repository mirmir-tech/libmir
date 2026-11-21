//! GGUF tensor names mapped to the `SafeTensors` names the binding layer
//! reads. GGUF names its tensors by role (`gguf-py` `TENSOR_NAMES`), the same
//! for every architecture, so the mapping is one table of roles, not of models.

/// Whole-model tensors.
const GLOBAL: [(&str, &str); 3] = [
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
];

/// Per-block tensors: `blk.N.<role>` → `model.layers.N.<name>`.
const BLOCK: [(&str, &str); 30] = [
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
    ("post_ffw_norm", "post_feedforward_layernorm"),
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
    ("ffn_gate_up_exps", "mlp.experts.gate_up_proj"),
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
];

/// In a block with both output norms (Gemma's sandwich), `ffn_norm` is the
/// norm before the feed-forward rather than after attention.
pub const SANDWICH_FEED_FORWARD_NORM: (&str, &str) = ("ffn_norm", "pre_feedforward_layernorm");

/// Roles whose `SafeTensors` name carries no `.weight` suffix.
const BARE: [(&str, &str); 2] = [
    ("attn_sinks.weight", "self_attn.sinks"),
    ("exp_probs_b.bias", "mlp.gate.e_score_correction_bias"),
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
    }
}
