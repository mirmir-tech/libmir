//! Gemma 4's per-layer inputs at the decoder boundary: the per-layer
//! embedding table, the projection of the token embeddings and its norm.

use models::{layout::DecoderConfig, semantic::SemanticModelSpec};

use super::{ModelSource, tensor};
use crate::{
    CudaBackend, CudaTensorSet, Error, Result,
    backend::{
        CheckpointProjectionWeight as ProjectionWeight, PerLayerInputTensors,
        PerLayerInputsTemplate,
    },
    checkpoint::config::to_f32,
};

/// The model's per-layer inputs, or `None` for a decoder without them.
pub(super) fn template(
    backend: &CudaBackend,
    (decoder, semantic): (&DecoderConfig, &SemanticModelSpec),
    tensors: &CudaTensorSet,
    source: &ModelSource<'_>,
) -> Result<Option<PerLayerInputsTemplate>> {
    let (spec, bindings) = match (semantic.decoder.per_layer_input, source.per_layer) {
        (Some(spec), Some(bindings)) => (spec, bindings),
        (None, None) => return Ok(None),
        _ => {
            return Err(Error::UnsupportedDecoderLayer(
                "per-layer inputs and their tensors disagree".into(),
            ));
        },
    };
    // Every token id indexes the per-layer table.
    if spec.vocab_size != decoder.vocab_size {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA needs a per-layer vocabulary of the model's {} tokens, not {}",
            decoder.vocab_size, spec.vocab_size
        )));
    }
    // The projection norm is the model's RMS norm, as the final norm.
    let norm = &semantic.decoder.final_norm;
    if norm.weight_offset != 0.0 {
        tensors.add_bf16(&bindings.projection_norm.source, norm.weight_offset)?;
    }
    let embedding = ProjectionWeight::load_binding_prepared(backend, tensors, bindings.embedding)?;
    let geometry = [decoder.vocab_size, decoder.num_hidden_layers, spec.width, decoder.hidden_size];
    Ok(Some(PerLayerInputsTemplate::new(
        PerLayerInputTensors {
            embedding,
            model_projection: ProjectionWeight::load_binding_prepared(
                backend,
                tensors,
                bindings.model_projection,
            )?,
            projection_norm: tensor(tensors, &bindings.projection_norm.source)?.clone(),
        },
        geometry,
        to_f32("RMS norm epsilon", norm.epsilon)?,
    )?))
}
