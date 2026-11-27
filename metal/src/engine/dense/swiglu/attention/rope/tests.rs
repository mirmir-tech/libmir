use models::layout::DecoderConfig;
use serde_json::json;

use super::{super::super::config::DenseSwiGluLayerConfig, rope_layout};
use crate::engine::{Array, Result, Stream};

/// A rotary attention factor scales the rotated pairs; a partial rotation's
/// pass-through dimensions keep their values.
#[test]
fn attention_factor_scales_only_rotated_dimensions() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 32, "vocab_size": 8,
        "rms_norm_eps": 1e-6
    }))?;
    let base = crate::engine::dense::swiglu::tests::layer_config(&decoder)?;
    let config = DenseSwiGluLayerConfig {
        rope_dimensions: 8,
        rope_attention_factor: 1.5,
        ..base
    };
    let plain = DenseSwiGluLayerConfig { rope_attention_factor: 1.0, ..config };
    let values = (0..2 * 3 * 16_u16).map(|index| f32::from(index % 7) - 3.0).collect::<Vec<_>>();
    // `[rows, sequence, heads, head_dim]` with three tokens.
    let input = Array::from_f32(&values, &[1, 3, 2, 16])?;
    let scaled = rope_layout(&input, None, config, 5, &stream)?.to_vec_f32(&stream)?;
    let reference = rope_layout(&input, None, plain, 5, &stream)?.to_vec_f32(&stream)?;
    for (index, (scaled, reference)) in scaled.iter().zip(&reference).enumerate() {
        let factor = if index % 16 < 8 {
            1.5
        } else {
            1.0
        };
        assert!((scaled - factor * reference).abs() < 1e-5, "dimension {}: {scaled}", index % 16);
    }
    Ok(())
}
