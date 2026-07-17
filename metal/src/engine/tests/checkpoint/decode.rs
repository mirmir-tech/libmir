use models::layout::{DecoderConfig, ModelLayout};

use super::*;
use crate::engine::hybrid_moe::{HybridMoeLayer, HybridMoeLayerConfig};

const EXPECTED_LAYERS: [u64; 30] = [
    0x69be_169d_89e4_8a1f,
    0x7ff8_b034_8309_602b,
    0xc8ac_29da_33a2_de23,
    0x741f_ba2b_a723_bba9,
    0x07f8_9bbe_0d06_c568,
    0x026b_c02a_d5f2_4f7d,
    0xf759_e7a6_f1cc_885b,
    0x3c19_149f_6f74_45ea,
    0x5924_cbaf_d68d_9f87,
    0x1d8d_4297_86fd_9ff2,
    0xaee2_6a31_395a_b9d9,
    0x7d04_7ded_b9ca_4ac4,
    0x1861_c99d_0d76_c44f,
    0xbb9b_c892_8bc6_89f6,
    0x9456_e954_c0ea_459e,
    0x4edf_48e4_5917_b133,
    0xa775_7146_8a67_c9d7,
    0x78c6_1eb2_34f7_ab51,
    0x4ad7_feff_a420_aaff,
    0xea82_85d0_0c1f_d8b6,
    0x6187_c9a2_0932_1266,
    0x93a8_7747_f631_59a4,
    0x99b8_90f6_12e6_ca79,
    0x0619_0f89_1e62_fa5e,
    0x2e61_2a34_763a_4b92,
    0x1bd1_f0d4_bd6b_1a51,
    0x7ad9_fc9f_20a5_a376,
    0xc209_7158_a062_ac8c,
    0x759f_60d1_d944_727b,
    0x710b_bdb4_cffb_a6da,
];

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_decode_after_128_token_prefill() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let scale = decoder.hidden_size.to_string().parse::<f32>()?.sqrt();
    let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
    let mut caches = Vec::with_capacity(decoder.num_hidden_layers);
    for index in 0..decoder.num_hidden_layers {
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        layers.push(HybridMoeLayer::load(&tensors, config, &stream)?);
        caches.push(KvCache::new_with_window(256, config.max_context)?);
    }

    let prompt = (1_000..1_127).collect::<Vec<_>>();
    let hidden = embedding
        .lookup(&Array::from_u32(&prompt, &[1, i32::try_from(prompt.len())?])?, &stream)?
        .multiply_scalar(scale, &stream)?;
    prefill(&layers, &mut caches, hidden, &stream)?;
    let last_prompt = embedding
        .lookup(&Array::from_u32(&[1_127], &[1, 1])?, &stream)?
        .multiply_scalar(scale, &stream)?;
    drop(decode_layers(&layers, &mut caches, last_prompt, 127, &stream)?);
    let mut hidden = embedding
        .lookup(&Array::from_u32(&[236_761], &[1, 1])?, &stream)?
        .multiply_scalar(scale, &stream)?;

    for (index, (layer, cache)) in layers.iter().zip(&mut caches).enumerate() {
        if index == 17 {
            let (query, rotated_query) = layer.query_for_test(&hidden, 128, &stream)?;
            assert_eq!(fingerprint(&query.to_vec_f32_on_stream(&stream)?), 0xe21e_4739_6b8f_72ec);
            assert_eq!(
                fingerprint(&rotated_query.to_vec_f32_on_stream(&stream)?),
                0x70d9_54cd_f0cc_818e
            );
            let mut snapshot = cache.snapshot_at(128)?;
            let attention =
                layer.attention_residual_for_test(&hidden, &mut snapshot, 128, false, &stream)?;
            assert_eq!(
                fingerprint(&attention.to_vec_f32_on_stream(&stream)?),
                0xbbe7_b361_9631_6b71
            );
            let feed_forward = layer.feed_forward_for_test(&attention, &stream)?;
            assert_eq!(
                fingerprint(&feed_forward.to_vec_f32_on_stream(&stream)?),
                0xb342_62d6_dfc1_c09c
            );
        }
        hidden = layer.forward_decode(&hidden, Some(cache), 128, false, &stream)?;
        hidden.async_eval()?;
        stream.synchronize()?;
        assert_eq!(
            fingerprint(&hidden.to_vec_f32_on_stream(&stream)?),
            EXPECTED_LAYERS[index],
            "layer {index}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_layer_one_attention_after_128_token_prefill() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let scale = decoder.hidden_size.to_string().parse::<f32>()?.sqrt();
    let mut layers = Vec::with_capacity(2);
    let mut caches = Vec::with_capacity(2);
    for index in 0..2 {
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        layers.push(HybridMoeLayer::load(&tensors, config, &stream)?);
        caches.push(KvCache::new_with_window(256, config.max_context)?);
    }
    let prompt = (1_000..1_127).collect::<Vec<_>>();
    let hidden = embedding
        .lookup(&Array::from_u32(&prompt, &[1, i32::try_from(prompt.len())?])?, &stream)?
        .multiply_scalar(scale, &stream)?;
    let hidden = layers[0].forward_decode(&hidden, Some(&mut caches[0]), 0, true, &stream)?;
    assert_eq!(fingerprint(&hidden.to_vec_f32_on_stream(&stream)?), 0x2f85_f9e8_f0dd_faf1);
    let (keys, values) = layers[1].key_value_for_test(&hidden, 0, &stream)?;
    assert_eq!(fingerprint(&keys.to_vec_f32_on_stream(&stream)?), 0xb1c6_fe70_bb2e_bf3f);
    assert_eq!(fingerprint(&values.to_vec_f32_on_stream(&stream)?), 0x2936_31ea_baf7_c99e);
    let hidden = layers[1].forward_decode(&hidden, Some(&mut caches[1]), 0, true, &stream)?;
    hidden.async_eval()?;
    stream.synchronize()?;
    let last_prompt = embedding
        .lookup(&Array::from_u32(&[1_127], &[1, 1])?, &stream)?
        .multiply_scalar(scale, &stream)?;
    drop(decode_layers(&layers, &mut caches, last_prompt, 127, &stream)?);
    let mut hidden = embedding
        .lookup(&Array::from_u32(&[236_761], &[1, 1])?, &stream)?
        .multiply_scalar(scale, &stream)?;
    hidden = layers[0].forward_decode(&hidden, Some(&mut caches[0]), 128, false, &stream)?;
    let (query, rotated_query) = layers[1].query_for_test(&hidden, 128, &stream)?;
    assert_eq!(fingerprint(&query.to_vec_f32_on_stream(&stream)?), 0x8645_c0c6_b8e2_c856);
    assert_eq!(
        fingerprint(&rotated_query.to_vec_f32_on_stream(&stream)?),
        0xf52b_625c_8c41_1e09
    );
    let attention =
        layers[1].attention_residual_for_test(&hidden, &mut caches[1], 128, false, &stream)?;
    assert_eq!(fingerprint(&attention.to_vec_f32_on_stream(&stream)?), 0x3433_da4e_e31e_a7b1);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_layer_zero_attention_during_127_token_prefill() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let config = HybridMoeLayerConfig::from_decoder(0, &decoder, 64)?;
    let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
    let input = embedding
        .lookup(&Array::from_u32(&(1_000..1_127).collect::<Vec<_>>(), &[1, 127])?, &stream)?
        .multiply_scalar(decoder.hidden_size.to_string().parse::<f32>()?.sqrt(), &stream)?;
    let attention = layer.attention_residual_for_test(
        &input,
        &mut KvCache::new_with_window(256, config.max_context)?,
        0,
        true,
        &stream,
    )?;
    assert_eq!(fingerprint(&attention.to_vec_f32_on_stream(&stream)?), 0x3f5d_46ff_d385_1d03);
    let (dense, routing, experts, feed_forward) =
        layer.feed_forward_components_for_test(&attention, &stream)?;
    assert_eq!(fingerprint(&dense.to_vec_f32_on_stream(&stream)?), 0xd44f_1db3_e7ea_fe75);
    let indices = routing
        .indices
        .to_vec_u32_on_stream(&stream)?
        .into_iter()
        .map(|index| Ok(f32::from(u16::try_from(index)?)))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(fingerprint(&indices), 0xcc54_29d7_fa6b_4ac6);
    assert_eq!(
        fingerprint(&routing.weights.to_vec_f32_on_stream(&stream)?),
        0xeccd_db0d_8bd3_9ad3
    );
    assert_eq!(fingerprint(&experts.to_vec_f32_on_stream(&stream)?), 0x5b0c_e543_6591_615f);
    assert_eq!(fingerprint(&feed_forward.to_vec_f32_on_stream(&stream)?), 0x271a_9e8d_9ba4_2113);
    Ok(())
}

fn prefill(
    layers: &[HybridMoeLayer],
    caches: &mut [KvCache],
    mut hidden: Array,
    stream: &Stream,
) -> Result<()> {
    for (layer, cache) in layers.iter().zip(caches) {
        hidden = layer.forward_decode(&hidden, Some(cache), 0, true, stream)?;
    }
    hidden.async_eval()?;
    stream.synchronize()
}

fn decode_layers(
    layers: &[HybridMoeLayer],
    caches: &mut [KvCache],
    mut hidden: Array,
    position: i32,
    stream: &Stream,
) -> Result<Array> {
    for (layer, cache) in layers.iter().zip(caches) {
        hidden = layer.forward_decode(&hidden, Some(cache), position, false, stream)?;
    }
    Ok(hidden)
}

fn fingerprint(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}
