use models::layout::{DecoderConfig, ModelLayout};

use super::*;
use crate::engine::hybrid_moe::{HybridMoeLayer, HybridMoeLayerConfig};

#[test]
#[ignore = "loads real model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_prefill_through_first_global_layer() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let mut hidden = embedding.lookup(&Array::from_u32(&[1_000, 1_001], &[1, 2])?, &stream)?;
    hidden =
        hidden.multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;

    for (index, expected) in expected_layers().iter().enumerate() {
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
        let mut cache = KvCache::new_with_window(16, config.max_context)?;
        hidden = layer.forward_decode(&hidden, Some(&mut cache), 0, true, &stream)?;
        hidden.async_eval(&stream)?;
        stream.synchronize()?;
        let values = hidden.to_vec_f32(&stream)?;
        assert_prefix(&values, expected, 0.002);
        if index < 2 {
            assert_signature(&values, expected_signatures()[index]);
        }
        if index == 1 {
            assert_samples(&values)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_layer_zero_components() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let input = embedding
        .lookup(&Array::from_u32(&[1_000, 1_001], &[1, 2])?, &stream)?
        .multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;
    assert_eq!(fingerprint(&input.to_vec_f32(&stream)?), 0xc809_b31e_15b3_6ebf);
    let config = HybridMoeLayerConfig::from_decoder(0, &decoder, 64)?;
    let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
    let mut cache = KvCache::new_with_window(16, config.max_context)?;
    let attention = layer.attention_residual_for_test(&input, &mut cache, 0, true, &stream)?;
    let attention_values = attention.to_vec_f32(&stream)?;
    assert_prefix(
        &attention_values,
        &[-0.906_25, -2.562_5, 0.052_734_375, 1.101_562_5, 1.375, -1.015_625, -1.218_75, 1.328_125],
        0.002,
    );
    assert_prefix(
        &attention_values[2_816..],
        &[
            -1.570_312_5, 1.984_375, -0.207_031_25, 0.730_468_75, -0.582_031_25, -1.953_125,
            -2.359_375, 0.468_75,
        ],
        0.002,
    );
    assert_signature(
        &attention_values,
        (481.648_437_5, 360_910.588_308_744_13, 1_960_905.497_802_734_4),
    );
    assert_eq!(fingerprint(&attention_values), 0xf742_8866_7c6f_f48e);
    let (normalized, scores) = layer.router_scores_for_test(&attention, &stream)?;
    assert_eq!(fingerprint(&normalized.to_vec_f32(&stream)?), 0x92b5_25c9_2256_5adc);
    assert_eq!(fingerprint(&scores.to_vec_f32(&stream)?), 0xb9c2_16d0_e6b5_90dc);

    let mut routing_cache = KvCache::new_with_window(16, config.max_context)?;
    let routing_input =
        layer.attention_residual_for_test(&input, &mut routing_cache, 0, true, &stream)?;
    let routing = layer.routing_for_test(&routing_input, &stream)?;
    let mut indices = routing.indices.to_vec_u32(&stream)?;
    indices[..8].sort_unstable();
    indices[8..].sort_unstable();
    assert_eq!(indices, vec![12, 26, 48, 49, 60, 64, 88, 111, 12, 48, 49, 57, 60, 64, 102, 123]);
    let weights = routing.weights.to_vec_f32(&stream)?;
    assert_eq!(fingerprint(&weights[..8]), 0x9317_8c1d_5079_b75a);
    assert_prefix(
        &weights,
        &[
            0.033_203_125, 0.043_457_03, 0.043_701_172, 0.072_753_906, 0.090_820_31, 0.172_851_56,
            0.241_210_94, 0.292_968_75,
        ],
        0.002,
    );

    let feed_forward = layer.feed_forward_for_test(&routing_input, &stream)?;
    assert_prefix(
        &feed_forward.to_vec_f32(&stream)?,
        &[
            0.126_953_13, 1.492_187_5, -0.679_687_5, 8.937_5, -0.029_052_734, 0.726_562_5,
            0.033_447_266, -0.414_062_5,
        ],
        0.002,
    );
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_proportional_rope_frequencies() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let config = HybridMoeLayerConfig::from_decoder(5, &decoder, 64)?;
    let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
    let frequencies = layer
        .rope_frequencies_for_test()
        .ok_or_else(|| Error::InvalidModel("missing proportional RoPE frequencies".into()))?;
    assert_eq!(fingerprint(&frequencies.to_vec_f32(&stream)?), 0xffcd_79f9_8a3d_335d);
    Ok(())
}

fn fingerprint(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn assert_signature(values: &[f32], expected: (f64, f64, f64)) {
    let signature = values.iter().enumerate().fold(
        (0.0, 0.0, 0.0),
        |(sum, square_sum, weighted_sum), (index, value)| {
            let value = f64::from(*value);
            let weight = f64::from(u32::try_from(index + 1).unwrap_or(u32::MAX));
            (
                sum + value,
                value.mul_add(value, square_sum),
                value.mul_add(weight, weighted_sum),
            )
        },
    );
    assert!((signature.0 - expected.0).abs() < 1.0e-4, "sum: {signature:?} != {expected:?}");
    assert!(
        (signature.1 - expected.1).abs() < 1.0e-4,
        "square sum: {signature:?} != {expected:?}"
    );
    assert!(
        (signature.2 - expected.2).abs() < 1.0e-3,
        "weighted sum: {signature:?} != {expected:?}"
    );
}

fn expected_signatures() -> [(f64, f64, f64); 2] {
    [
        (16.060_817_718_506, 16_966.759_872_965_995, 99_381.427_185_058_6),
        (-240.899_496_078_491, 14_972.890_492_737_435, -842_184.023_734_092_7),
    ]
}

fn assert_samples(values: &[f32]) -> Result<()> {
    let indexes = [8, 31, 127, 511, 1_023, 2_047, 2_815, 2_816, 3_000, 4_095, 5_000, 5_631];
    let expected = [
        -0.141_601_56, -0.018_432_617, -5.437_5, -0.001_388_549_8, -2.062_5, 0.064_453_125,
        0.064_453_125, 0.004_760_742, 0.011_474_609, 0.019_287_11, 0.115_722_656, 0.023_803_711,
    ];
    for (index, expected) in indexes.into_iter().zip(expected) {
        let actual = *values.get(index).ok_or(Error::ShapeOverflow)?;
        assert!((actual - expected).abs() < 0.002, "sample {index}: {actual} != {expected}");
    }
    Ok(())
}

fn expected_layers() -> [[f32; 8]; 6] {
    [
        [
            -0.054_931_64, -0.075_195_31, -0.043_945_312, 0.707_031_25, 0.094_726_56,
            -0.020_263_672, -0.083_496_094, 0.064_453_125,
        ],
        [
            -0.066_406_25, -0.128_906_25, -0.037_353_516, -0.851_562_5, 0.001_983_642_6,
            0.009_338_379, -0.043_212_89, -0.010_314_941,
        ],
        [
            -0.028_320_312, -0.083_007_81, -0.010_437_012, -1.273_437_5, -0.005_310_058_6,
            0.007_293_701, -0.012_878_418, -0.005_126_953,
        ],
        [
            0.886_718_75, -0.137_695_31, -0.008_056_641, -1.382_812_5, -0.008_056_641,
            0.025_268_555, -0.003_845_214_8, -0.000_724_792_5,
        ],
        [
            0.067_382_81, -0.128_906_25, -0.006_530_761_7, 0.158_203_13, 0.010_742_187_5,
            0.032_714_844, -0.008_300_781, -0.003_540_039,
        ],
        [
            0.765_625, -0.138_671_88, -0.011_474_609, -0.239_257_81, -0.014_465_332, 0.332_031_25,
            -0.012_451_172, -0.004_028_320_3,
        ],
    ]
}
