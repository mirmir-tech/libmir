//! Gemma 4's E models: layers sharing an earlier layer's keys and values, and
//! per-layer inputs.

use serde_json::json;

use super::{
    super::{LayerCacheShape, PerLayerInputSpec},
    catalog, *,
};

fn spec() -> Result<SemanticModelSpec> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 32, "sliding_window": 16,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8
    }))?;
    SemanticModelSpec::discover(&decoder, &catalog(&["model.layers.0.input_layernorm.weight"]))
}

fn source(spec: &SemanticModelSpec, index: usize) -> Result<Option<usize>> {
    Ok(spec
        .decoder
        .layer(index)?
        .mixer
        .softmax_attention()
        .and_then(|a| a.key_value_source))
}

#[test]
fn shared_layers_and_per_layer_inputs_reach_the_spec() -> Result<()> {
    let spec = spec()?;
    assert_eq!(
        (0..4).map(|index| source(&spec, index)).collect::<Result<Vec<_>>>()?,
        [None, None, Some(0), Some(1)]
    );
    assert_eq!(
        spec.decoder.per_layer_input,
        Some(PerLayerInputSpec { width: 8, vocab_size: 32 })
    );
    Ok(())
}

/// A sharing layer attends over its source's cache and keeps none.
#[test]
fn only_source_layers_cache_keys_and_values() -> Result<()> {
    let spec = spec()?;
    let shapes = spec
        .decoder
        .layers
        .iter()
        .map(DecoderLayerSpec::cache_shape)
        .collect::<Vec<_>>();
    let cached = Some(LayerCacheShape { heads: 2, key_width: 16, value_width: 16 });
    assert_eq!(shapes, [cached, cached, None, None]);
    Ok(())
}

#[test]
fn a_layer_cannot_share_keys_of_another_geometry() -> Result<()> {
    let mut spec = spec()?;
    // Layer 3 (full attention) pointed at layer 0 (a sliding window).
    if let MixerSpec::SoftmaxAttention(attention) = &mut spec.decoder.layers[3].mixer {
        attention.key_value_source = Some(0);
    }
    let error = spec.to_toml().err().map(|error| error.to_string());
    assert!(error.is_some_and(|error| error.contains("another geometry")));
    Ok(())
}
