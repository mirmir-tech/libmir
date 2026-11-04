//! Logical shapes of multi-token prediction module tensors.

use super::layer;
use crate::{semantic::SemanticModelSpec, weights::binding::PredictionTensorRole};

pub(super) fn logical(
    spec: &SemanticModelSpec,
    depth: usize,
    role: &PredictionTensorRole,
) -> Option<Vec<usize>> {
    let module = spec.decoder.prediction_layers.get(depth)?;
    let hidden = spec.decoder.hidden_size;
    match role {
        PredictionTensorRole::EmbeddingNorm
        | PredictionTensorRole::HiddenNorm
        | PredictionTensorRole::FinalNorm => Some(vec![hidden]),
        // `[embedding | hidden]` concatenated back to the hidden width.
        PredictionTensorRole::Fusion => Some(vec![hidden, hidden.checked_mul(2)?]),
        PredictionTensorRole::CacheScale { .. } => Some(vec![1]),
        PredictionTensorRole::Layer { tensor } => layer(spec, module, tensor),
    }
}
