//! Shapes of the per-layer input tensors (Gemma 4's E models).

use super::super::{PerLayerEmbeddingRole, PerLayerInputRole};
use crate::semantic::SemanticModelSpec;

/// `[vocab, layers * width]`, `[layers * width, hidden]` and `[width]`.
pub(super) fn embedding(
    spec: &SemanticModelSpec,
    role: PerLayerEmbeddingRole,
) -> Option<Vec<usize>> {
    let input = spec.decoder.per_layer_input?;
    let all = spec.decoder.layers.len().checked_mul(input.width)?;
    Some(match role {
        PerLayerEmbeddingRole::Embedding => vec![input.vocab_size, all],
        PerLayerEmbeddingRole::ModelProjection => vec![all, spec.decoder.hidden_size],
        PerLayerEmbeddingRole::ProjectionNorm => vec![input.width],
    })
}

/// `[width, hidden]`, `[hidden, width]` and `[hidden]`.
pub(super) fn input(spec: &SemanticModelSpec, role: PerLayerInputRole) -> Option<Vec<usize>> {
    let width = spec.decoder.per_layer_input?.width;
    let hidden = spec.decoder.hidden_size;
    Some(match role {
        PerLayerInputRole::Gate => vec![width, hidden],
        PerLayerInputRole::Projection => vec![hidden, width],
        PerLayerInputRole::OutputNorm => vec![hidden],
    })
}
