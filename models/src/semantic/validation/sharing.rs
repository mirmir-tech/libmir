//! Layers that reuse an earlier layer's keys and values, and the per-layer
//! inputs of Gemma 4's E models.

use super::{super::DecoderSpec, invalid};
use crate::{error::Result, semantic::MixerSpec};

pub(super) fn validate(decoder: &DecoderSpec) -> Result<()> {
    if decoder
        .per_layer_input
        .is_some_and(|input| input.width == 0 || input.vocab_size == 0)
    {
        return Err(invalid("per-layer inputs need a positive width and vocabulary"));
    }
    for layer in &decoder.layers {
        let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
            continue;
        };
        let Some(source) = attention.key_value_source else {
            continue;
        };
        let Some(MixerSpec::SoftmaxAttention(origin)) = decoder
            .layers
            .get(source)
            .filter(|_| source < layer.index)
            .map(|layer| &layer.mixer)
        else {
            return Err(invalid(format!(
                "layer {} shares keys and values with a layer that is not earlier softmax attention",
                layer.index
            )));
        };
        // The keys are attended as stored: the source must cache what this
        // layer would, rotated as its queries are.
        if origin.key_value_source.is_some()
            || origin.key_value_heads != attention.key_value_heads
            || origin.head_dim != attention.head_dim
            || origin.window != attention.window
            || origin.position != attention.position
        {
            return Err(invalid(format!(
                "layer {} shares keys and values with layer {source} of another geometry",
                layer.index
            )));
        }
    }
    Ok(())
}
