use models::layout::{DecoderConfig, ModelLayout};

use super::*;
use crate::engine::hybrid_moe::{HybridMoeLayer, HybridMoeLayerConfig, HybridMoeModel};

#[test]
#[ignore = "loads all model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn executes_real_gemma_model_decode_logits() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let model = HybridMoeModel::load(&tensors, &decoder, 64, 16, &stream)?;
    let mut cache = model.new_cache(&stream)?;
    let ids = Array::from_u32(&[1], &[1, 1])?;
    let logits = model.forward_decode(&ids, &mut cache, 0, &stream)?;
    logits.async_eval()?;
    stream.synchronize()?;

    assert_eq!(model.layer_count(), 30);
    assert_eq!(logits.shape()?, vec![1, 1, 262_144]);
    let expected = [-15.25, 5.156_25, -12.625, -14.875, -16.125, -16.125, -17.25, -12.75];
    let values = logits.to_vec_f32_on_stream(&stream)?;
    assert_prefix(&values, &expected, 0.26);
    let mut best = (0_u32, f32::NEG_INFINITY);
    for (index, value) in values.iter().copied().enumerate() {
        if value > best.1 {
            best = (u32::try_from(index)?, value);
        }
    }
    assert_eq!(best.0, 236_795);

    cache.reset()?;
    let greedy_logits = model.forward_greedy_decode(&ids, &mut cache, 0, &stream)?;
    assert_eq!(greedy_logits.argmax_u32(&stream)?, best.0);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn compares_real_gemma_first_layer_from_embedding() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let config = HybridMoeLayerConfig::from_decoder(0, &decoder, 64)?;
    let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let ids = Array::from_u32(&[1], &[1, 1])?;
    let input = embedding
        .lookup(&ids, &stream)?
        .multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;
    let output = layer.forward_uncached_decode(&input, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;

    assert_eq!(output.dtype()?, Dtype::Bfloat16);
    let expected = [
        -0.021_484_375, 0.006_591_797, 0.302_734_38, 0.071_289_06, 0.010_559_08, -0.007_690_43,
        -0.087_402_34, 0.053_466_8,
    ];
    assert_prefix(&output.to_vec_f32_on_stream(&stream)?, &expected, 0.002);
    Ok(())
}

#[test]
#[ignore = "loads real model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_two_token_prefill_after_layer_one() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let mut hidden = embedding.lookup(&Array::from_u32(&[1_000, 1_001], &[1, 2])?, &stream)?;
    hidden =
        hidden.multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;
    for index in 0..=1 {
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
        let mut cache = KvCache::new_with_window(16, config.max_context)?;
        hidden = layer.forward_decode(&hidden, Some(&mut cache), 0, true, &stream)?;
    }
    hidden.async_eval()?;
    stream.synchronize()?;

    let expected = [
        -0.066_406_25, -0.128_906_25, -0.037_353_516, -0.851_562_5, 0.001_983_642_6, 0.009_338_379,
        -0.043_212_89, -0.010_314_941,
    ];
    assert_prefix(&hidden.to_vec_f32_on_stream(&stream)?, &expected, 0.002);
    Ok(())
}

#[test]
#[ignore = "loads real model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_two_token_prefill_after_last_sliding_layer() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let mut hidden = embedding.lookup(&Array::from_u32(&[1_000, 1_001], &[1, 2])?, &stream)?;
    hidden =
        hidden.multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;
    for index in 0..=4 {
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
        let mut cache = KvCache::new_with_window(16, config.max_context)?;
        hidden = layer.forward_decode(&hidden, Some(&mut cache), 0, true, &stream)?;
    }
    hidden.async_eval()?;
    stream.synchronize()?;

    let expected = [
        0.067_382_81, -0.128_906_25, -0.006_530_761_7, 0.158_203_13, 0.010_742_187_5,
        0.032_714_844, -0.008_300_781, -0.003_540_039,
    ];
    assert_prefix(&hidden.to_vec_f32_on_stream(&stream)?, &expected, 0.002);
    Ok(())
}

#[test]
#[ignore = "loads real model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_two_token_prefill_after_first_global_attention() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let mut hidden = embedding.lookup(&Array::from_u32(&[1_000, 1_001], &[1, 2])?, &stream)?;
    hidden =
        hidden.multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;
    for index in 0..=5 {
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
        let mut cache = KvCache::new_with_window(16, config.max_context)?;
        hidden = layer.forward_decode(&hidden, Some(&mut cache), 0, true, &stream)?;
    }
    hidden.async_eval()?;
    stream.synchronize()?;

    let expected = [
        0.765_625, -0.138_671_88, -0.011_474_609, -0.239_257_81, -0.014_465_332, 0.332_031_25,
        -0.012_451_172, -0.004_028_320_3,
    ];
    assert_prefix(&hidden.to_vec_f32_on_stream(&stream)?, &expected, 0.002);
    Ok(())
}

#[test]
#[ignore = "loads all model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_token_decode_after_chunked_prefill() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let model = HybridMoeModel::load(&tensors, &decoder, 64, 16, &stream)?;
    let mut cache = model.new_cache(&stream)?;
    let tokens = [1_u32, 2, 3];
    let mut token_logits = None;
    for (position, token) in tokens.iter().copied().enumerate() {
        let position = i32::try_from(position)?;
        token_logits = Some(model.forward_decode(
            &Array::from_u32(&[token], &[1, 1])?,
            &mut cache,
            position,
            &stream,
        )?);
    }
    let token_logits =
        token_logits.ok_or_else(|| Error::InvalidModel("test tokens missing".into()))?;
    token_logits.async_eval()?;
    stream.synchronize()?;
    let expected = token_logits.to_vec_f32_on_stream(&stream)?;

    cache.reset()?;
    let prefix = Array::from_u32(&tokens[..2], &[1, 2])?;
    let prefetched = model.forward_prefill(&prefix, &mut cache, 0, &stream)?;
    prefetched.async_eval()?;
    stream.synchronize()?;
    let actual =
        model.forward_decode(&Array::from_u32(&[tokens[2]], &[1, 1])?, &mut cache, 2, &stream)?;
    actual.async_eval()?;
    stream.synchronize()?;
    assert_eq!(cache.cached_tokens()?, tokens.len());
    assert_prefix(&actual.to_vec_f32_on_stream(&stream)?, &expected[..64], 0.002);
    Ok(())
}
