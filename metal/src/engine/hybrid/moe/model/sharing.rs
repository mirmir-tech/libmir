//! Gemma 4's E models in the hybrid mixture-of-experts runtime: layers that
//! attend over an earlier layer's keys and values, and every layer's per-layer
//! input.

use std::{collections::BTreeSet, sync::Arc};

use models::{
    semantic::{DecoderLayerSpec, DecoderSpec},
    weights::WeightBindingPlan,
};

use super::HybridMoeModel;
use crate::engine::{
    Array, ModelTensors, Result, Stream,
    hybrid::moe::{
        HybridMoeLayer, HybridMoeLayerConfig,
        sharing::{PerLayerGate, PerLayerInputs, StepShare},
        weights::Publish,
    },
    lowering::LayerLowering,
};

/// The model's per-layer inputs and the step slot its layers share.
#[derive(Debug)]
pub(in crate::engine::hybrid::moe) struct ModelSharing {
    inputs: Option<PerLayerInputs>,
    step: Arc<StepShare>,
    /// The layers whose keys and values later layers read.
    sources: BTreeSet<usize>,
}

impl ModelSharing {
    /// `None` for a decoder without per-layer inputs or shared keys.
    pub(super) fn load(
        tensors: &ModelTensors,
        spec: &DecoderSpec,
        bindings: &WeightBindingPlan,
        stream: &Stream,
    ) -> Result<Option<Self>> {
        let sources = spec
            .layers
            .iter()
            .filter_map(|layer| layer.mixer.softmax_attention()?.key_value_source)
            .collect::<BTreeSet<_>>();
        if spec.per_layer_input.is_none() && sources.is_empty() {
            return Ok(None);
        }
        let inputs = match (spec.per_layer_input, bindings.per_layer_embedding()?) {
            (Some(input), Some(embedding)) => Some(PerLayerInputs::load(
                tensors,
                embedding,
                [spec.layers.len(), input.width, spec.hidden_size],
                spec.final_norm.epsilon.to_string().parse()?,
                stream,
            )?),
            (None, None) => None,
            _ => {
                return Err(crate::engine::Error::InvalidModel(
                    "per-layer inputs and their tensors disagree".into(),
                ));
            },
        };
        Ok(Some(Self {
            inputs,
            step: Arc::new(StepShare::new(spec.layers.len())),
            sources,
        }))
    }

    /// A dense GELU layer: reading its source's keys and values, or projecting
    /// its own (published when later layers read them); with its per-layer
    /// input gate.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_dense_layer(
        sharing: Option<&Self>,
        tensors: &ModelTensors,
        bindings: &WeightBindingPlan,
        layer: &DecoderLayerSpec,
        lowered: LayerLowering,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<HybridMoeLayer> {
        let index = layer.index;
        let source =
            layer.mixer.softmax_attention().and_then(|attention| attention.key_value_source);
        let loaded = match (sharing, source) {
            (Some(sharing), Some(source)) => HybridMoeLayer::load_shared_dense_bindings(
                tensors,
                &bindings.dense_decoder_layer_sharing_key_values(index)?,
                lowered,
                layer,
                (Arc::clone(&sharing.step), source),
                config,
                stream,
            )?,
            _ => HybridMoeLayer::load_dense_bindings(
                tensors,
                &bindings.dense_decoder_layer(index)?,
                lowered,
                layer,
                sharing
                    .filter(|sharing| sharing.sources.contains(&index))
                    .map(|sharing| Publish {
                        step: Arc::clone(&sharing.step),
                        layer: index,
                    }),
                config,
                stream,
            )?,
        };
        let gate = match (sharing, bindings.per_layer_input(index)?) {
            (Some(sharing), Some(gate)) => {
                Some(PerLayerGate::load(tensors, gate, index, Arc::clone(&sharing.step), stream)?)
            },
            _ => None,
        };
        Ok(loaded.with_per_layer_gate(gate))
    }
}

impl HybridMoeModel {
    /// Computes the step's per-layer inputs from `token_ids` and their
    /// `scaled` token embeddings, before the layers run.
    pub(in crate::engine::hybrid::moe) fn prepare_step(
        &self,
        token_ids: &Array,
        scaled: &Array,
        stream: &Stream,
    ) -> Result<()> {
        if let Some(ModelSharing { inputs: Some(inputs), step, .. }) = &self.sharing {
            step.set_per_layer(inputs.compute(token_ids, scaled, stream)?);
        }
        Ok(())
    }
}
