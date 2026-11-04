use super::QkNormalization;
use crate::weights::TensorCatalog;

/// Q/K normalization of one attention layer, from its checkpoint tensors.
/// Value normalization has no tensor, so only a model spec declares it.
pub(super) fn qk_normalization(tensors: &TensorCatalog, index: usize) -> QkNormalization {
    let has = |name: &str| {
        layer_roots(index)
            .into_iter()
            .any(|root| tensors.contains(&format!("{root}.{name}")))
    };
    if has("self_attn.q_norm.weight") && has("self_attn.k_norm.weight") {
        QkNormalization::QueryKeyRms
    } else {
        QkNormalization::None
    }
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
