//! Binding grammar of multi-token prediction modules: the fusion inputs, an
//! attention block and a routed block with hybrid-layer grammar, and the
//! output norm. Tensors under `mtp.layers.` must all have a typed role when
//! the model declares prediction modules; otherwise they stay auxiliary.

use super::{
    invalid,
    state_space::{feed_forward, mixer},
};
use crate::{
    error::Result,
    semantic::{MixerSpec, SemanticModelSpec},
    weights::binding::{LayerTensorRole, LogicalTensorRole, PredictionTensorRole, TensorBinding},
};

const FUSION: [PredictionTensorRole; 4] = [
    PredictionTensorRole::EmbeddingNorm,
    PredictionTensorRole::HiddenNorm,
    PredictionTensorRole::Fusion,
    PredictionTensorRole::FinalNorm,
];

pub(super) fn validate(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    if spec.decoder.prediction_layers.is_empty() {
        return Ok(());
    }
    unrecognized(tensors)?;
    for module in &spec.decoder.prediction_layers {
        let depth = module.index;
        let has = |tensor| has_prediction(tensors, depth, tensor);
        let has_layer = |tensor| has(PredictionTensorRole::Layer { tensor });
        let complete = FUSION.into_iter().all(has)
            && matches!(module.mixer, MixerSpec::SoftmaxAttention(_))
            && mixer(module, tensors, &has_layer)?
            && feed_forward(module, &has_layer);
        if !complete {
            return Err(invalid(format!(
                "prediction module {depth} has an incomplete binding grammar"
            )));
        }
    }
    Ok(())
}

fn has_prediction(tensors: &[TensorBinding], depth: usize, tensor: PredictionTensorRole) -> bool {
    let role = LogicalTensorRole::Prediction { depth, tensor };
    tensors.binary_search_by(|binding| binding.role.cmp(&role)).is_ok()
}

fn unrecognized(tensors: &[TensorBinding]) -> Result<()> {
    let unknown = tensors.iter().find(|binding| match &binding.role {
        LogicalTensorRole::Prediction {
            tensor:
                PredictionTensorRole::Layer {
                    tensor: LayerTensorRole::Auxiliary { .. },
                },
            ..
        } => true,
        LogicalTensorRole::Auxiliary { path } => path.starts_with("mtp.layers."),
        _ => false,
    });
    unknown.map_or(Ok(()), |binding| {
        Err(invalid(format!(
            "prediction checkpoint tensor {} has no role in its module",
            binding.source
        )))
    })
}
