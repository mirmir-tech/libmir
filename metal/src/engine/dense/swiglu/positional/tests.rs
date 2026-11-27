//! Chunked attention against each query attending alone to the keys of its
//! own chunk.

use models::layout::DecoderConfig;
use serde_json::json;

use super::{super::config::DenseSwiGluLayerConfig, ViewOrigin, attend};
use crate::engine::{Array, KvContext, Result, Stream};

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            f32::from(u16::try_from((index * 37 + seed * 11) % 101).unwrap_or(0)) / 50.0 - 1.0
        })
        .collect()
}

#[test]
fn chunked_queries_see_only_their_chunk() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64, "vocab_size": 8,
        "rms_norm_eps": 1e-6
    }))?;
    let config = DenseSwiGluLayerConfig {
        window: std::num::NonZeroUsize::new(4)
            .map(|chunk| models::semantic::AttentionWindow::Chunked { chunk }),
        ..crate::engine::dense::swiglu::tests::layer_config(&decoder)?
    };
    // Ten keys; six queries at positions 4..10 span chunks 1 and 2.
    let (total, sequence) = (10_usize, 6_usize);
    let keys = Array::from_f32(&values(2 * total * 16, 1), &[1, 2, 10, 16])?;
    let key_values = Array::from_f32(&values(2 * total * 16, 2), &[1, 2, 10, 16])?;
    let queries = Array::from_f32(&values(4 * sequence * 16, 3), &[1, 4, 6, 16])?;
    let context = KvContext {
        keys: keys.astype_like(&keys, &stream)?,
        values: key_values.astype_like(&key_values, &stream)?,
        paged: None,
        mask: None,
    };
    let output = attend(&queries, &context, config, ViewOrigin { newest: 9, last: 9 }, &stream)?
        .ok_or(crate::engine::Error::InvalidModel("expected chunked attention".into()))?
        .to_vec_f32(&stream)?;
    for row in 0..sequence {
        let position = total - sequence + row;
        let start = position - position % 4;
        let query = queries.slice(&[0, 0, row, 0], &[1, 4, row + 1, 16], &stream)?;
        let own =
            |array: &Array| array.slice(&[0, 0, start, 0], &[1, 2, position + 1, 16], &stream);
        let expected = query
            .scaled_dot_product_attention(
                &own(&keys)?,
                &own(&key_values)?,
                config.attention_scale,
                false,
                &stream,
            )?
            .to_vec_f32(&stream)?;
        for head in 0..4 {
            for lane in 0..16 {
                let actual = output[(head * sequence + row) * 16 + lane];
                let wanted = expected[head * 16 + lane];
                assert!(
                    (actual - wanted).abs() < 1e-4,
                    "row {row} head {head}: {actual} != {wanted}"
                );
            }
        }
    }
    // A single decode query at position 9 attends to keys 8 and 9 only.
    let last = queries.slice(&[0, 0, 5, 0], &[1, 4, 6, 16], &stream)?;
    let single = attend(&last, &context, config, ViewOrigin { newest: 9, last: 9 }, &stream)?
        .ok_or(crate::engine::Error::InvalidModel("expected chunked decode".into()))?
        .to_vec_f32(&stream)?;
    let row = |head: usize| &output[(head * sequence + 5) * 16..(head * sequence + 6) * 16];
    for head in 0..4 {
        for (actual, wanted) in single[head * 16..(head + 1) * 16].iter().zip(row(head)) {
            assert!((actual - wanted).abs() < 1e-4);
        }
    }
    // The same query over a six-slot decode ring holding positions 4..10 as
    // [8, 9, 4, 5, 6, 7], its newest token in slot 1.
    let ring = |array: &Array| -> Result<Array> {
        let tail = array.slice(&[0, 0, 8, 0], &[1, 2, 10, 16], &stream)?;
        let head = array.slice(&[0, 0, 4, 0], &[1, 2, 8, 16], &stream)?;
        Array::concatenate(&[&tail, &head], 2, &stream)
    };
    let ringed = KvContext {
        keys: ring(&keys)?,
        values: ring(&key_values)?,
        paged: None,
        mask: None,
    };
    let from_ring = attend(&last, &ringed, config, ViewOrigin { newest: 1, last: 9 }, &stream)?
        .ok_or(crate::engine::Error::InvalidModel("expected chunked ring decode".into()))?
        .to_vec_f32(&stream)?;
    for (actual, wanted) in from_ring.iter().zip(&single) {
        assert!((actual - wanted).abs() < 1e-4, "ring {actual} != {wanted}");
    }
    Ok(())
}

/// A chunk-wide window cache fed by prefill chunks and decode tokens across
/// chunk boundaries matches each query attending to its chunk's history.
#[test]
fn window_cache_of_one_chunk_matches_full_history() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64, "vocab_size": 8,
        "rms_norm_eps": 1e-6
    }))?;
    let chunk = 8;
    let config = DenseSwiGluLayerConfig {
        window: std::num::NonZeroUsize::new(chunk)
            .map(|chunk| models::semantic::AttentionWindow::Chunked { chunk }),
        ..crate::engine::dense::swiglu::tests::layer_config(&decoder)?
    };
    let total = 30;
    let keys = Array::from_f32(&values(2 * total * 16, 4), &[1, 2, 30, 16])?;
    let key_values = Array::from_f32(&values(2 * total * 16, 5), &[1, 2, 30, 16])?;
    let queries = Array::from_f32(&values(4 * total * 16, 6), &[1, 4, 30, 16])?;
    let span = |array: &Array, heads: usize, from: usize, to: usize| {
        array.slice(&[0, 0, from, 0], &[1, heads, to, 16], &stream)
    };
    let mut cache = crate::engine::KvCache::new_with_window(4, Some(chunk))?;
    let mut position = 0;
    for tokens in [5, 6, 3, 9, 1, 1, 1, 1, 1, 1, 1] {
        let context = cache.update_for_attention_mode(
            &span(&keys, 2, position, position + tokens)?,
            &span(&key_values, 2, position, position + tokens)?,
            &stream,
            usize::MAX,
            crate::engine::PagedContextMode::View,
        )?;
        let query = span(&queries, 4, position, position + tokens)?;
        let last = position + tokens - 1;
        let newest = cache.window_newest_slot().ok_or(crate::engine::Error::ShapeOverflow)?;
        let output =
            match attend(&query, &context, config, ViewOrigin { newest, last }, &stream)? {
                Some(output) => output,
                None => match context.mask.as_ref() {
                    Some(mask) => query.masked_scaled_dot_product_attention(
                        &context.keys,
                        &context.values,
                        config.attention_scale,
                        mask,
                        &stream,
                    )?,
                    None => query.scaled_dot_product_attention(
                        &context.keys,
                        &context.values,
                        config.attention_scale,
                        tokens > 1,
                        &stream,
                    )?,
                },
            }
            .to_vec_f32(&stream)?;
        for row in 0..tokens {
            let at = position + row;
            let start = at - at % chunk;
            let expected = span(&queries, 4, at, at + 1)?
                .scaled_dot_product_attention(
                    &span(&keys, 2, start, at + 1)?,
                    &span(&key_values, 2, start, at + 1)?,
                    config.attention_scale,
                    false,
                    &stream,
                )?
                .to_vec_f32(&stream)?;
            for head in 0..4 {
                for lane in 0..16 {
                    let actual = output[(head * tokens + row) * 16 + lane];
                    let wanted = expected[head * 16 + lane];
                    assert!((actual - wanted).abs() < 1e-4, "position {at}: {actual} != {wanted}");
                }
            }
        }
        position += tokens;
    }
    Ok(())
}
