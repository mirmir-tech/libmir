//! Nemotron-H (`nemotron_h`) names: each GGUF block holds one Mamba-2,
//! attention or MLP block under the checkpoint's `backbone.` namespace.
//!
//! llama.cpp's converter (`conversion/mamba.py`, `Mamba2Model`) stores
//! `A = -exp(A_log)` and `D` as `[heads, 1]`, the convolution without its unit
//! axis and the gated norm as `[groups, inner / groups]`; the shapes are
//! restored here and `STATE_SPACE_LAYOUT` tells the runtime about `A`.

use crate::weights::TensorInfo;

/// (GGUF block tensor, checkpoint name under `backbone.layers.N.`)
const BLOCK: [(&str, &str); 15] = [
    ("attn_norm.weight", "norm.weight"),
    ("ssm_in.weight", "mixer.in_proj.weight"),
    ("ssm_conv1d.weight", "mixer.conv1d.weight"),
    ("ssm_conv1d.bias", "mixer.conv1d.bias"),
    ("ssm_dt.bias", "mixer.dt_bias"),
    ("ssm_a", "mixer.A_log"),
    ("ssm_d", "mixer.D"),
    ("ssm_norm.weight", "mixer.norm.weight"),
    ("ssm_out.weight", "mixer.out_proj.weight"),
    ("attn_q.weight", "mixer.q_proj.weight"),
    ("attn_k.weight", "mixer.k_proj.weight"),
    ("attn_v.weight", "mixer.v_proj.weight"),
    ("attn_output.weight", "mixer.o_proj.weight"),
    ("ffn_up.weight", "mixer.up_proj.weight"),
    ("ffn_down.weight", "mixer.down_proj.weight"),
];

const GLOBAL: [(&str, &str); 3] = [
    ("token_embd.weight", "backbone.embeddings.weight"),
    ("output_norm.weight", "backbone.norm_f.weight"),
    ("output.weight", "lm_head.weight"),
];

/// Renames a Nemotron-H catalog and restores the checkpoint shapes.
pub(super) fn map(tensors: &mut [TensorInfo]) {
    for tensor in tensors {
        let Some(name) = name(&tensor.name) else {
            continue;
        };
        tensor.shape = match (name.rsplit_once('.').map(|(_, leaf)| leaf), tensor.shape.as_slice())
        {
            // `[channels, kernel]` back to `[channels, 1, kernel]`.
            (Some("weight"), &[channels, kernel]) if name.ends_with("conv1d.weight") => {
                vec![channels, 1, kernel]
            },
            (Some("A_log" | "D"), &[heads, 1]) => vec![heads],
            (Some("weight"), &[groups, width]) if name.ends_with("mixer.norm.weight") => {
                vec![groups * width]
            },
            _ => tensor.shape.clone(),
        };
        tensor.name = name;
    }
}

fn name(gguf: &str) -> Option<String> {
    if let Some((_, name)) = GLOBAL.iter().find(|(from, _)| *from == gguf) {
        return Some((*name).to_owned());
    }
    let (layer, rest) = gguf.strip_prefix("blk.")?.split_once('.')?;
    let (_, name) = BLOCK.iter().find(|(from, _)| *from == rest)?;
    Some(format!("backbone.layers.{layer}.{name}"))
}
