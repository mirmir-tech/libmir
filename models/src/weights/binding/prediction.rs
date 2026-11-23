//! Typed binding view of multi-token prediction modules.

use super::{
    LogicalTensorRole, MixedDecoderLayerBindings, PredictionTensorRole,
    StateSpaceHybridLayerBindings, TensorBinding, WeightBindingPlan, slot::LayerSlot,
    view::sources,
};
use crate::error::{ModelsError, Result};

/// Bindings of one multi-token prediction module.
///
/// The fusion of the next token's embedding with the decoder's last hidden
/// state, the attention and routed block, and the output norm before the
/// shared head.
#[derive(Debug, Clone, Copy)]
pub struct PredictionLayerBindings<'a> {
    pub embedding_norm: &'a TensorBinding,
    pub hidden_norm: &'a TensorBinding,
    /// `[hidden, 2 · hidden]`, applied to `[embedding | hidden]`.
    pub fusion: &'a TensorBinding,
    pub layer: StateSpaceHybridLayerBindings<'a>,
    pub final_norm: &'a TensorBinding,
}

/// A prediction module whose layer is a hybrid decoder layer (Qwen3.5's
/// `mtp`): the same fusion inputs and output norm around one full-attention
/// layer with a dense or shared-routed feed-forward.
#[derive(Debug, Clone, Copy)]
pub struct MixedPredictionLayerBindings<'a> {
    pub embedding_norm: &'a TensorBinding,
    pub hidden_norm: &'a TensorBinding,
    /// `[hidden, 2 · hidden]`, applied to `[embedding | hidden]`.
    pub fusion: &'a TensorBinding,
    pub layer: MixedDecoderLayerBindings<'a>,
    pub final_norm: &'a TensorBinding,
}

impl WeightBindingPlan {
    fn prediction_tensor(
        &self,
        depth: usize,
        tensor: PredictionTensorRole,
    ) -> Result<&TensorBinding> {
        let role = LogicalTensorRole::Prediction { depth, tensor };
        self.binding(&role).ok_or_else(|| {
            ModelsError::InvalidConfig(format!(
                "invalid prediction module {depth}: unbound role {role:?}"
            ))
        })
    }

    /// Whether the checkpoint binds prediction module `depth`.
    #[must_use]
    pub fn has_prediction_module(&self, depth: usize) -> bool {
        self.binding(&LogicalTensorRole::Prediction {
            depth,
            tensor: PredictionTensorRole::Fusion,
        })
        .is_some()
    }

    pub fn mixed_prediction_layer(&self, depth: usize) -> Result<MixedPredictionLayerBindings<'_>> {
        let get = |tensor| self.prediction_tensor(depth, tensor);
        Ok(MixedPredictionLayerBindings {
            embedding_norm: get(PredictionTensorRole::EmbeddingNorm)?,
            hidden_norm: get(PredictionTensorRole::HiddenNorm)?,
            fusion: get(PredictionTensorRole::Fusion)?,
            layer: self.mixed_layer(LayerSlot::Prediction(depth))?,
            final_norm: get(PredictionTensorRole::FinalNorm)?,
        })
    }

    pub fn prediction_layer(&self, depth: usize) -> Result<PredictionLayerBindings<'_>> {
        let get = |tensor| self.prediction_tensor(depth, tensor);
        Ok(PredictionLayerBindings {
            embedding_norm: get(PredictionTensorRole::EmbeddingNorm)?,
            hidden_norm: get(PredictionTensorRole::HiddenNorm)?,
            fusion: get(PredictionTensorRole::Fusion)?,
            layer: self.hybrid_layer(LayerSlot::Prediction(depth))?,
            final_norm: get(PredictionTensorRole::FinalNorm)?,
        })
    }
}

impl<'a> PredictionLayerBindings<'a> {
    /// Every checkpoint tensor and companion this module reads.
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = self.layer.physical_sources();
        bindings.extend(sources([
            self.embedding_norm,
            self.hidden_norm,
            self.fusion,
            self.final_norm,
        ]));
        bindings
    }
}
