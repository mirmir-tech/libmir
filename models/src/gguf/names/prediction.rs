//! Qwen3.5's multi-token prediction module in GGUF: llama.cpp appends it as
//! block `L + d` after the `L` decoder blocks (`block_count` counts both),
//! an ordinary full-attention block with the fusion inputs and output norm
//! under `nextn.`. The checkpoint names the same tensors `mtp.*`.

use super::GgufHeader;
use crate::weights::TensorInfo;

/// Architectures whose modules use Qwen3.5's `mtp` namespace.
const ARCHITECTURES: [&str; 2] = ["qwen35", "qwen35moe"];

/// `blk.N.nextn.<role>` → `mtp.<name>`, for the one module this namespace
/// holds.
const FUSION: [(&str, &str); 4] = [
    ("eh_proj.weight", "fc.weight"),
    ("enorm.weight", "pre_fc_norm_embedding.weight"),
    ("hnorm.weight", "pre_fc_norm_hidden.weight"),
    ("shared_head_norm.weight", "norm.weight"),
];

/// Renames the prediction blocks of an already mapped catalog: decoder-named
/// `model.layers.{L + d}.*` become `mtp.layers.{d}.*`, and the fusion inputs
/// move to the namespace root. Leaves other architectures alone.
pub(super) fn map(header: &GgufHeader, tensors: &mut [TensorInfo]) {
    let Some(architecture) = header.architecture().filter(|name| ARCHITECTURES.contains(name))
    else {
        return;
    };
    let key = |name: &str| header.unsigned(&format!("{architecture}.{name}"));
    let (Some(modules), Some(blocks)) = (key("nextn_predict_layers"), key("block_count")) else {
        return;
    };
    let Some(decoder) = blocks.checked_sub(modules).filter(|_| modules == 1) else {
        return;
    };
    for tensor in tensors {
        if let Some(name) = rename(&tensor.name, decoder) {
            tensor.name = name;
        }
    }
}

fn rename(name: &str, decoder: u64) -> Option<String> {
    if let Some(role) = name.strip_prefix(&format!("blk.{decoder}.nextn.")) {
        let (_, mapped) = FUSION.iter().find(|(from, _)| *from == role)?;
        return Some(format!("mtp.{mapped}"));
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let depth = layer.parse::<u64>().ok()?.checked_sub(decoder)?;
    Some(format!("mtp.layers.{depth}.{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::rename;

    #[test]
    fn moves_the_appended_block_into_the_module_namespace() {
        assert_eq!(rename("blk.64.nextn.eh_proj.weight", 64).as_deref(), Some("mtp.fc.weight"));
        assert_eq!(
            rename("blk.64.nextn.shared_head_norm.weight", 64).as_deref(),
            Some("mtp.norm.weight")
        );
        assert_eq!(
            rename("model.layers.64.self_attn.q_proj.weight", 64).as_deref(),
            Some("mtp.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(rename("model.layers.63.self_attn.q_proj.weight", 64), None);
        assert_eq!(rename("blk.64.nextn.embed_tokens.weight", 64), None);
    }
}
