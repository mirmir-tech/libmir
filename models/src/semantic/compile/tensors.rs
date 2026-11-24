use super::QkNormalization;
use crate::{layout::DecoderConfig, semantic::NormPlacement, weights::TensorCatalog};

/// Q/K normalization of one attention layer, from its checkpoint tensors.
/// Value normalization has no tensor; import defaults or a model spec declare
/// it. Q/K norms from their tensors: per head, or across heads when the query
/// norm spans the whole `query_width` projection (`OLMo` 2).
pub(super) fn qk_normalization(
    tensors: &TensorCatalog,
    index: usize,
    query_width: usize,
) -> QkNormalization {
    let find = |name: &str| {
        layer_roots(index)
            .into_iter()
            .find_map(|root| tensors.get(&format!("{root}.{name}")))
    };
    match (find("self_attn.q_norm.weight"), find("self_attn.k_norm.weight")) {
        (Some(query), Some(_)) if query.shape.as_slice() == [query_width] && query_width > 0 => {
            QkNormalization::QueryKeyRmsAcrossHeads
        },
        (Some(_), Some(_)) => QkNormalization::QueryKeyRms,
        _ => QkNormalization::None,
    }
}

/// Post-norm blocks have output norms and no input norm (`OLMo` 2);
/// parallel blocks only the input norm (Cohere).
pub(super) fn norm_placement(
    decoder: &DecoderConfig,
    tensors: &TensorCatalog,
    index: usize,
) -> NormPlacement {
    let has = |name: &str| {
        layer_roots(index)
            .into_iter()
            .any(|root| tensors.contains(&format!("{root}.{name}")))
    };
    if decoder.parallel_block {
        NormPlacement::Parallel
    } else if !has("input_layernorm.weight")
        && has("post_attention_layernorm.weight")
        && has("post_feedforward_layernorm.weight")
    {
        NormPlacement::Post
    } else {
        NormPlacement::Pre
    }
}

pub(super) fn has_shared_expert_gate(tensors: &TensorCatalog, index: usize) -> bool {
    layer_roots(index)
        .into_iter()
        .any(|root| tensors.contains(&format!("{root}.mlp.shared_expert_gate.weight")))
}

pub(super) fn has_attention_sinks(tensors: &TensorCatalog, index: usize) -> bool {
    layer_roots(index)
        .into_iter()
        .any(|root| tensors.contains(&format!("{root}.self_attn.sinks")))
}

fn layer_roots(index: usize) -> [String; 4] {
    [
        format!("model.layers.{index}"),
        format!("layers.{index}"),
        format!("language_model.model.layers.{index}"),
        format!("model.language_model.layers.{index}"),
    ]
}
