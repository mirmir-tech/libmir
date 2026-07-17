use std::{env, path::PathBuf};

use super::*;

mod decode;
mod embedding;
mod layer;
mod long_prefill;
mod model;
mod output;
mod prefill;

const LAYER: &str = "language_model.model.layers.0";

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn projects_with_real_gemma_q_weight() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let quantized = quantized(&tensors, &format!("{LAYER}.self_attn.q_proj"))?;
    let input = Array::from_f32(&vec![0.25; 2_816], &[1, 1, 2_816])?;
    let output = input.quantized_matmul(&quantized, true, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    assert!(!tensors.is_empty());
    assert_eq!(tensors.file_count(), 4);
    assert_eq!(tensors.len(), 1_339);
    assert_eq!(quantized.weight.dtype()?, Dtype::Uint32);
    assert_eq!(output.shape()?, vec![1, 1, 4_096]);
    let values = output.to_vec_f32()?;
    let expected = [
        0.311_720_85, -0.176_174_16, -0.121_078_97, -0.502_513_9, 0.128_841_4, 0.342_447_28,
        -0.117_445_47, -0.282_201_3,
    ];
    assert_prefix(&values, &expected, 1.0e-5);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn loads_real_gemma_quantized_embedding() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let ids = Array::from_u32(&[1], &[1, 1])?;
    let output = embedding.lookup(&ids, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;

    assert_eq!(embedding.bits(), 8);
    assert_eq!(output.shape()?, vec![1, 1, 2_816]);
    let expected = [
        -0.011_230_469, 0.071_777_344, 0.048_339_844, -0.052_734_375, 0.007_293_701, 0.011_230_469,
        -0.016_235_352, 0.019_653_32,
    ];
    assert_prefix(&output.to_vec_f32_on_stream(&stream)?, &expected, 1.0e-6);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn executes_real_gemma_layer_zero_attention() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let input = Array::from_f32(&gemma_input(), &[1, 1, 2_816])?;
    let output = hybrid_moe_attention(&tensors, &input, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;

    assert_eq!(output.shape()?, vec![1, 1, 2_816]);
    let expected = [
        1.008_520_8, -6.367_076, 1.805_791_1, 0.184_957_98, -2.887_073, 0.695_407_87, -3.078_099_5,
        0.637_955_9,
    ];
    assert_prefix(&output.to_vec_f32()?, &expected, 1.0e-4);
    Ok(())
}

fn hybrid_moe_attention(tensors: &ModelTensors, input: &Array, stream: &Stream) -> Result<Array> {
    let attention = format!("{LAYER}.self_attn");
    let input_norm = tensors.get(&format!("{LAYER}.input_layernorm.weight"))?;
    let hidden = input.rms_norm(&input_norm, 1.0e-6, stream)?;

    let queries = project(&hidden, tensors, &format!("{attention}.q_proj"), 16, stream)?;
    let q_norm = tensors.get(&format!("{attention}.q_norm.weight"))?;
    let queries = queries.rms_norm(&q_norm, 1.0e-6, stream)?;
    let queries = attention_layout(&queries, stream)?;

    let keys = project(&hidden, tensors, &format!("{attention}.k_proj"), 8, stream)?;
    let k_norm = tensors.get(&format!("{attention}.k_norm.weight"))?;
    let keys = keys.rms_norm(&k_norm, 1.0e-6, stream)?;
    let keys = attention_layout(&keys, stream)?;

    let values = project(&hidden, tensors, &format!("{attention}.v_proj"), 8, stream)?;
    let values = values.rms_norm_unit(1.0e-6, stream)?;
    let values = values.transpose(&[0, 2, 1, 3], stream)?;

    let output = queries.scaled_dot_product_attention(&keys, &values, 1.0, false, stream)?;
    let output = output.transpose(&[0, 2, 1, 3], stream)?;
    let output = output.reshape(&[1, 1, 4_096], stream)?;
    output.quantized_matmul(&quantized(tensors, &format!("{attention}.o_proj"))?, true, stream)
}

fn project(
    input: &Array,
    tensors: &ModelTensors,
    prefix: &str,
    heads: i32,
    stream: &Stream,
) -> Result<Array> {
    input
        .quantized_matmul(&quantized(tensors, prefix)?, true, stream)?
        .reshape(&[1, 1, heads, 256], stream)
}

fn attention_layout(input: &Array, stream: &Stream) -> Result<Array> {
    input.transpose(&[0, 2, 1, 3], stream)?.rope(
        RopeOptions {
            dimensions: 256,
            traditional: false,
            base: Some(10_000.0),
            scale: 1.0,
            offset: 0,
        },
        stream,
    )
}

fn quantized(tensors: &ModelTensors, prefix: &str) -> Result<QuantizedArrays> {
    QuantizedArrays::new(
        tensors.get(&format!("{prefix}.weight"))?,
        tensors.get(&format!("{prefix}.scales"))?,
        tensors.get(&format!("{prefix}.biases"))?,
        64,
        8,
    )
}

fn load_model() -> Result<(ModelTensors, Stream)> {
    let root = model_root()?;
    let tensors = ModelTensors::load(root, &Stream::new_cpu()?)?;
    tensors.evaluate()?;
    let stream = Stream::new_gpu()?;
    let _configured_wired_limit = configure_recommended_wired_limit()?;
    Ok((tensors, stream))
}

fn model_root() -> Result<PathBuf> {
    env::var_os("MIRMIR_BENCH_MODEL")
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or_else(|| Error::InvalidModel("set MIRMIR_BENCH_MODEL or MODEL".into()))
}

fn gemma_input() -> Vec<f32> {
    (0_u8..31)
        .cycle()
        .take(2_816)
        .map(|index| (f32::from(index) - 15.0) / 16.0)
        .collect()
}

fn gemma_input_shifted() -> Vec<f32> {
    (1_u8..32)
        .map(|index| (f32::from(index % 31) - 15.0) / 16.0)
        .cycle()
        .take(2_816)
        .collect()
}

fn assert_prefix(actual: &[f32], expected: &[f32], tolerance: f32) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < tolerance, "{actual} != {expected}");
    }
}
