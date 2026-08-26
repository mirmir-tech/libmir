#![allow(clippy::print_stdout)]

use std::{hint::black_box, time::Instant};

use super::*;
use crate::engine::hybrid_moe::{HybridMoeLayer, HybridMoeLayerConfig};

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn executes_real_gemma_layer_zero() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let input = Array::from_f32(&gemma_input(), &[1, 1, 2_816])?;
    let layer = HybridMoeLayer::load(&tensors, config(), &stream)?;
    let output = layer.forward_uncached_decode(&input, &stream)?;
    output.async_eval(&stream)?;
    stream.synchronize()?;

    assert_eq!(output.shape()?, vec![1, 1, 2_816]);
    let expected = [
        -0.044_613_752, 0.162_483_96, -0.117_699_42, -0.089_524_07, 0.019_990_986, -0.049_411_934,
        -0.122_311_16, -0.097_376_33,
    ];
    assert_prefix(&output.to_vec_f32(&stream)?, &expected, 2.0e-4);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn executes_real_gemma_layer_zero_with_kv_cache() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let layer = HybridMoeLayer::load(&tensors, config(), &stream)?;
    let mut cache = KvCache::new(2)?;
    let first = Array::from_f32(&gemma_input(), &[1, 1, 2_816])?;
    let second = Array::from_f32(&gemma_input_shifted(), &[1, 1, 2_816])?;
    let _first = layer.forward_decode(&first, Some(&mut cache), 0, false, &stream)?;
    let output = layer.forward_decode(&second, Some(&mut cache), 1, false, &stream)?;
    output.async_eval(&stream)?;
    stream.synchronize()?;

    assert_eq!(cache.offset()?, 2);
    let expected = [
        0.036_895_838, 0.019_147, -0.186_714_07, 0.663_706_7, -0.084_519_78, -0.022_803_692,
        -0.197_952_1, -0.064_229_69,
    ];
    assert_prefix(&output.to_vec_f32(&stream)?, &expected, 2.0e-4);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_layer_zero_causal_prefill() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let input = embedding
        .lookup(&Array::from_u32(&[1_000, 1_001], &[1, 2])?, &stream)?
        .multiply_scalar(2_816.0_f32.sqrt(), &stream)?;
    let layer = HybridMoeLayer::load(&tensors, config(), &stream)?;
    let mut cache = KvCache::new(16)?;
    let output = layer.forward_decode(&input, Some(&mut cache), 0, true, &stream)?;
    output.async_eval(&stream)?;
    stream.synchronize()?;

    let expected = [
        -0.054_931_64, -0.075_195_31, -0.043_945_312, 0.707_031_25, 0.094_726_56, -0.020_263_672,
        -0.083_496_094, 0.064_453_125, -1.046_875, -0.042_480_47, 0.087_890_625, 0.107_421_875,
        -0.164_062_5, 0.679_687_5, 0.012_084_961, -0.146_484_38,
    ];
    assert_prefix(&output.to_vec_f32(&stream)?, &expected, 0.002);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn executes_real_gemma_full_attention_layer() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let layer = HybridMoeLayer::load(&tensors, full_config(), &stream)?;
    let input = Array::from_f32(&gemma_input(), &[1, 1, 2_816])?;
    let output = layer.forward_uncached_decode(&input, &stream)?;
    output.async_eval(&stream)?;
    stream.synchronize()?;

    let expected = [
        -0.622_566_1, -0.519_252_9, -0.533_876_36, -0.323_585_5, -0.431_795_18, -0.612_654_6,
        -0.394_577_65, -0.314_409_26,
    ];
    assert_prefix(&output.to_vec_f32(&stream)?, &expected, 3.0e-4);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn paged_full_attention_matches_contiguous_cache() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let layer = HybridMoeLayer::load(&tensors, full_config(), &stream)?;
    let first = Array::from_f32(&gemma_input(), &[1, 1, 2_816])?;
    let second = Array::from_f32(&gemma_input_shifted(), &[1, 1, 2_816])?;
    let mut contiguous = KvCache::new(2)?;
    let mut paged = KvCache::new_paged(2, 16)?;
    drop(layer.forward_decode(&first, Some(&mut contiguous), 0, false, &stream)?);
    let expected = layer.forward_decode(&second, Some(&mut contiguous), 1, false, &stream)?;
    drop(layer.forward_decode(&first, Some(&mut paged), 0, false, &stream)?);
    let actual = layer.forward_decode(&second, Some(&mut paged), 1, false, &stream)?;
    actual.async_eval(&stream)?;
    stream.synchronize()?;
    let expected = expected.to_vec_f32(&stream)?;
    let actual = actual.to_vec_f32(&stream)?;
    assert_prefix(&actual, &expected[..64], 0.002);

    let third = Array::from_f32(&gemma_input(), &[1, 1, 2_816])?;
    let expected = layer.forward_decode(&third, Some(&mut contiguous), 2, false, &stream)?;
    expected.async_eval(&stream)?;
    stream.synchronize()?;
    let expected = expected.to_vec_f32(&stream)?;
    let mut prefix = gemma_input();
    prefix.extend(gemma_input_shifted());
    let prefix = Array::from_f32(&prefix, &[1, 2, 2_816])?;
    let mut prefetched = KvCache::new_paged(2, 16)?;
    drop(layer.forward_decode(&prefix, Some(&mut prefetched), 0, true, &stream)?);
    let actual = layer.forward_decode(&third, Some(&mut prefetched), 2, false, &stream)?;
    actual.async_eval(&stream)?;
    stream.synchronize()?;
    let actual = actual.to_vec_f32(&stream)?;
    assert_prefix(&actual, &expected[..64], 0.002);
    Ok(())
}

#[test]
#[ignore = "benchmark; set MIRMIR_BENCH_MODEL or MODEL"]
fn benchmarks_real_gemma_layer_zero() -> Result<()> {
    const ITERATIONS: u32 = 10;
    let (tensors, stream) = load_model()?;
    let layer = HybridMoeLayer::load(&tensors, config(), &stream)?;
    let reference = tensors.get("language_model.model.norm.weight")?;
    let input =
        Array::from_f32(&gemma_input(), &[1, 1, 2_816])?.astype_like(&reference, &stream)?;
    let warmup = layer.forward_uncached_decode(&input, &stream)?;
    warmup.async_eval(&stream)?;
    stream.synchronize()?;

    let started = Instant::now();
    for _iteration in 0..ITERATIONS {
        let output = layer.forward_uncached_decode(&input, &stream)?;
        output.async_eval(&stream)?;
        stream.synchronize()?;
        black_box(output);
    }
    let synchronized = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(ITERATIONS);

    let mut chained =
        Array::from_f32(&gemma_input(), &[1, 1, 2_816])?.astype_like(&reference, &stream)?;
    let started = Instant::now();
    for _iteration in 0..ITERATIONS {
        chained = layer.forward_uncached_decode(&chained, &stream)?;
    }
    chained.async_eval(&stream)?;
    stream.synchronize()?;
    black_box(chained);
    let pipelined = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(ITERATIONS);

    println!("native_gemma_layer_decode synchronized_ms={synchronized:.3}");
    println!("native_gemma_layer_decode pipelined_ms={pipelined:.3}");
    Ok(())
}

fn config() -> HybridMoeLayerConfig {
    HybridMoeLayerConfig {
        layer_index: 0,
        hidden_size: 2_816,
        attention_heads: 16,
        kv_heads: 8,
        head_dim: 256,
        rope_dimensions: 256,
        rope_base: 10_000.0,
        proportional_rope: false,
        use_k_eq_v: false,
        rms_norm_eps: 1.0e-6,
        top_k: 8,
        expert_count: 128,
        expert_intermediate: 2_816,
        group_size: 64,
        router_norm_scale: 0.018_844_46,
        max_context: None,
    }
}

fn full_config() -> HybridMoeLayerConfig {
    HybridMoeLayerConfig {
        layer_index: 5,
        hidden_size: 2_816,
        attention_heads: 16,
        kv_heads: 2,
        head_dim: 512,
        rope_dimensions: 128,
        rope_base: 1_000_000.0,
        proportional_rope: true,
        use_k_eq_v: true,
        rms_norm_eps: 1.0e-6,
        top_k: 8,
        expert_count: 128,
        expert_intermediate: 2_816,
        group_size: 64,
        router_norm_scale: 0.018_844_46,
        max_context: None,
    }
}
