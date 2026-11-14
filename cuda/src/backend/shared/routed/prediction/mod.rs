//! The multi-token prediction module of a state-space hybrid checkpoint.
//!
//! For position `i` the module reads the decoder's normalized output `h_i`
//! and the following token `t_{i+1}`:
//! `final_norm(layer(fusion([enorm(embed(t_{i+1})) | hnorm(h_i)])))`. The
//! layer is an attention mixer with its own K/V over the sequence and a
//! latent routed feed-forward; the shared output head turns the result into
//! logits for `t_{i+2}`.

mod execution;
mod verify;

use models::{semantic::SemanticModelSpec, weights::WeightBindingPlan};

pub(super) use self::execution::PredictionExecution;
use super::{SharedRoutedLayerTemplate, load::HybridLoad, required_norm};
use crate::{CudaAttentionHybridLayer, CudaTensor, CudaTensorDType, Error, Result};

#[derive(Clone, Debug)]
pub(super) struct PredictionTemplate {
    embedding_norm: CudaTensor,
    hidden_norm: CudaTensor,
    /// `[hidden, 2 · hidden]` over `[embedding | hidden]`.
    fusion: CudaTensor,
    layer: Box<CudaAttentionHybridLayer>,
    final_norm: CudaTensor,
    epsilon: f32,
    /// The module attention's K/V slot, after the decoder layers.
    index: usize,
}

impl PredictionTemplate {
    pub(super) fn load(
        load: &HybridLoad<'_>,
        semantic: &SemanticModelSpec,
        bindings: &WeightBindingPlan,
    ) -> Result<Self> {
        let spec = semantic.decoder.prediction_layers.first().ok_or_else(|| {
            Error::UnsupportedDecoderLayer("checkpoint has no multi-token prediction module".into())
        })?;
        let module = bindings.prediction_layer(0)?;
        let index = load.decoder.num_hidden_layers;
        let SharedRoutedLayerTemplate::Attention(layer) = load.layer(index, spec, module.layer)?
        else {
            return Err(Error::UnsupportedDecoderLayer(
                "multi-token prediction module mixer is not attention".into(),
            ));
        };
        let hidden = load.decoder.hidden_size;
        let norm = |name: &str| required_norm(load.tensors, name, hidden);
        let fusion = load
            .tensors
            .get(&module.fusion.source)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(module.fusion.source.clone()))?;
        let expected = vec![hidden, 2 * hidden];
        if fusion.shape() != expected {
            return Err(Error::InvalidQuantizedTensor {
                name: module.fusion.source.clone(),
                expected,
                actual: fusion.shape().to_vec(),
            });
        }
        if fusion.dtype() != CudaTensorDType::Bf16 {
            return Err(Error::DTypeMismatch {
                name: module.fusion.source.clone(),
                expected: "BF16",
            });
        }
        Ok(Self {
            embedding_norm: norm(&module.embedding_norm.source)?,
            hidden_norm: norm(&module.hidden_norm.source)?,
            fusion,
            layer,
            final_norm: norm(&module.final_norm.source)?,
            epsilon: load.decoder.rms_norm_eps.to_string().parse()?,
            index,
        })
    }

    pub(super) const fn index(&self) -> usize {
        self.index
    }

    pub(super) const fn layer(&self) -> &CudaAttentionHybridLayer {
        &self.layer
    }
}
