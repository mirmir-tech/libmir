use models::layout::{DecoderConfig, ModelLayout};

use super::*;
use crate::engine::hybrid_moe::{HybridMoeLayer, HybridMoeLayerConfig};

const PREFIX_TOKENS: usize = 2_047;
const LAYER_ZERO_DECODE: [f32; 8] = [
    0.106_933_594, -0.043_457_03, -0.204_101_56, 0.002_746_582, 0.072_265_625, 0.000_549_316,
    -0.045_654_297, -0.140_625,
];
const LAYER_ONE_ATTENTION: [f32; 8] = [
    0.142_578_13, -0.109_375, -0.201_171_88, -1.101_562_5, 0.066_406_25, 0.000_949_86,
    -0.054_931_64, -0.133_789_06,
];
const PREFIX_LAYER_ZERO: &[(usize, [f32; 8])] = &[
    (
        0,
        [
            -0.054_931_64, -0.077_636_72, -0.043_945_313, 0.703_125, 0.093_75, -0.019_531_25,
            -0.083_984_375, 0.064_453_125,
        ],
    ),
    (
        1_023,
        [
            0.055_419_922, 0.075_683_594, 0.010_253_906, -0.096_679_69, 0.097_656_25,
            -0.294_921_88, -0.084_472_656, 0.035_156_25,
        ],
    ),
    (
        1_024,
        [
            0.040_527_344, 0.118_652_344, -0.073_242_19, 0.222_656_25, -0.022_460_938,
            -0.337_890_63, -0.116_210_94, 0.028_808_594,
        ],
    ),
    (
        2_046,
        [
            -0.132_812_5, 0.219_726_56, -0.025_024_414, 0.007_690_43, 0.081_054_69, 0.106_445_31,
            -0.096_191_406, -0.113_281_25,
        ],
    ),
];

#[test]
#[ignore = "loads two real model layers; set MIRMIR_BENCH_MODEL or MODEL"]
fn matches_mlx_lm_at_long_prefill_entry_layers() -> Result<()> {
    let root = model_root()?;
    let decoder = DecoderConfig::from_layout(&ModelLayout::inspect(&root)?)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let scale = decoder.hidden_size.to_string().parse::<f32>()?.sqrt();
    let configs = (0..2)
        .map(|index| HybridMoeLayerConfig::from_decoder(index, &decoder, 64))
        .collect::<Result<Vec<_>>>()?;
    let layers = configs
        .iter()
        .map(|config| HybridMoeLayer::load(&tensors, *config, &stream))
        .collect::<Result<Vec<_>>>()?;
    let mut caches = configs
        .iter()
        .map(|config| KvCache::new_with_window(16, config.max_context))
        .collect::<Result<Vec<_>>>()?;
    let prefix = (0..PREFIX_TOKENS)
        .map(|token| u32::try_from(token + 1_000))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut hidden = embed(&embedding, &prefix, scale, &stream)?;
    for (index, (layer, cache)) in layers.iter().zip(&mut caches).enumerate() {
        hidden = layer.forward_decode(&hidden, Some(cache), 0, true, &stream)?;
        if index == 0 {
            assert_positions(&hidden.to_vec_f32_on_stream(&stream)?, decoder.hidden_size);
        }
    }

    hidden = embed(&embedding, &[3_047], scale, &stream)?;
    hidden = layers[0].forward_decode(&hidden, Some(&mut caches[0]), 2_047, false, &stream)?;
    assert_prefix(&hidden.to_vec_f32_on_stream(&stream)?, &LAYER_ZERO_DECODE, 1.0e-6);
    let mut cache = caches[1].snapshot_at(caches[1].offset()?)?;
    let attention =
        layers[1].attention_residual_for_test(&hidden, &mut cache, 2_047, false, &stream)?;
    assert_prefix(&attention.to_vec_f32_on_stream(&stream)?, &LAYER_ONE_ATTENTION, 0.002);
    Ok(())
}

fn assert_positions(values: &[f32], width: usize) {
    for (position, expected) in PREFIX_LAYER_ZERO {
        let start = position * width;
        assert_prefix(&values[start..start + expected.len()], expected, 1.0e-6);
    }
}

fn embed(
    embedding: &QuantizedEmbedding,
    tokens: &[u32],
    scale: f32,
    stream: &Stream,
) -> Result<Array> {
    let length = i32::try_from(tokens.len())?;
    embedding
        .lookup(&Array::from_u32(tokens, &[1, length])?, stream)?
        .multiply_scalar(scale, stream)
}
