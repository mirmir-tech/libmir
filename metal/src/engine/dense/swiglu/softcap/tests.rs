//! Capped attention against fused scaled dot-product attention, with a cap
//! far beyond every score so that it changes nothing.

use models::layout::DecoderConfig;
use serde_json::json;

use super::attend;
use crate::engine::{Array, KvCache, PagedContextMode, Result, Stream};

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            f32::from(u16::try_from((index * 37 + seed * 11) % 101).unwrap_or(0)) / 50.0 - 1.0
        })
        .collect()
}

#[test]
fn a_wide_cap_matches_fused_attention_over_grouped_heads() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64, "vocab_size": 8,
        "rms_norm_eps": 1e-6
    }))?;
    let config = crate::engine::dense::swiglu::tests::layer_config(&decoder)?;
    let mut cache = KvCache::new_paged(16, 16)?;
    // Three earlier tokens, then five queries attending causally.
    for (tokens, seed) in [(3_usize, 1_usize), (5, 2)] {
        let keys =
            Array::from_f32(&values(2 * tokens * 16, seed), &[1, 2, i32::try_from(tokens)?, 16])?;
        let new_values = Array::from_f32(
            &values(2 * tokens * 16, seed + 5),
            &[1, 2, i32::try_from(tokens)?, 16],
        )?;
        let context = cache.update_for_attention_mode(
            &keys,
            &new_values,
            &stream,
            0,
            PagedContextMode::View,
        )?;
        if tokens == 5 {
            let queries = Array::from_f32(&values(4 * 5 * 16, 9), &[1, 4, 5, 16])?;
            let capped =
                attend(&queries, &context, config, 1.0e6, true, &stream)?.to_vec_f32(&stream)?;
            let fused = queries
                .scaled_dot_product_attention(
                    &context.keys,
                    &context.values,
                    config.attention_scale,
                    true,
                    &stream,
                )?
                .to_vec_f32(&stream)?;
            for (capped, fused) in capped.iter().zip(&fused) {
                assert!((capped - fused).abs() <= 1.0e-4, "{capped} != {fused}");
            }
        }
    }
    Ok(())
}
