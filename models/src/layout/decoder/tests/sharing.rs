//! Gemma 4's E models: layers sharing keys and values, and per-layer inputs.

use serde_json::json;

use super::super::*;
use crate::error::Result;

/// Gemma 4 E4B's last layers read the keys and values of the last earlier
/// layer of their attention type, as MLX-LM's `previous_kvs`.
#[test]
fn shared_key_value_layers_read_the_last_earlier_layer_of_their_type() -> Result<()> {
    let mut types = vec!["sliding_attention"; 8];
    types[3] = "full_attention";
    types[7] = "full_attention";
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 8,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 32, "layer_types": types, "sliding_window": 16,
        "num_kv_shared_layers": 3, "hidden_size_per_layer_input": 8,
        "vocab_size_per_layer_input": 32
    }))?;
    let sources = (0..8).map(|index| decoder.key_value_source(index)).collect::<Vec<_>>();
    assert_eq!(sources, [None, None, None, None, None, Some(4), Some(4), Some(3)]);
    assert_eq!(decoder.per_layer_input, Some(PerLayerInputConfig { width: 8, vocab_size: 32 }));
    // A shared full-attention layer with no full-attention layer before it.
    let error = DecoderConfig::from_value(&json!({
        "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 32, "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 16, "num_kv_shared_layers": 1
    }))
    .err()
    .map(|error| error.to_string());
    assert!(error.is_some_and(|error| error.contains("no earlier layer")));
    Ok(())
}
