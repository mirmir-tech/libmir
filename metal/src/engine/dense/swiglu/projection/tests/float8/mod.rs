use std::{fs, path::Path};

use models::weights::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
    Float8ScaleGranularity, Float8ScaleMode, LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::*;

mod block_grid;
mod embedding;

#[test]
fn executes_direct_fp8_formats_on_metal() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-fp8-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    assert_eq!(tensors.get("weight")?.dtype()?, crate::engine::Dtype::Uint8);
    let stream = Stream::new_gpu()?;
    let input =
        Array::from_f32(&[1.0, 2.0], &[1, 2])?.astype(crate::engine::Dtype::Bfloat16, &stream)?;

    let multiplied = BoundLinear::load(&tensors, &binding(Float8ScaleMode::Multiplier), &stream)?;
    let output = multiplied.forward(&input, &stream)?;
    assert_eq!(output.dtype()?, crate::engine::Dtype::Bfloat16);
    assert_eq!(output.to_vec_f32(&stream)?, [11.0, -1.0]);
    let divided =
        BoundLinear::load(&tensors, &binding(Float8ScaleMode::InverseMultiplier), &stream)?;
    assert_eq!(divided.forward(&input, &stream)?.to_vec_f32(&stream)?, [3.5, -1.0]);
    for binding in [
        static_binding(Float8ParameterDType::F32, "input_scale"),
        static_binding(Float8ParameterDType::BF16, "input_scale_bf16"),
    ] {
        let static_linear = BoundLinear::load(&tensors, &binding, &stream)?;
        assert_eq!(static_linear.forward(&input, &stream)?.to_vec_f32(&stream)?, [11.0, -1.0]);
    }
    let scaled_e5m2 = BoundLinear::load(&tensors, &e5m2_binding(true), &stream)?;
    assert_eq!(scaled_e5m2.forward(&input, &stream)?.to_vec_f32(&stream)?, [11.0, -1.0]);
    let unscaled_e5m2 = BoundLinear::load(&tensors, &e5m2_binding(false), &stream)?;
    assert_eq!(unscaled_e5m2.forward(&input, &stream)?.to_vec_f32(&stream)?, [6.0, -1.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires MIRMIR_FP8_MODEL"]
fn checkpoint_layer_zero_matches_the_fp8_oracle() -> Result<()> {
    use models::{
        layout::{DecoderConfig, ModelLayout},
        semantic::SemanticModelSpec,
        weights::{TensorCatalog, WeightBindingPlan},
    };

    let root = std::env::var("MIRMIR_FP8_MODEL").map_err(|_| {
        crate::engine::Error::InvalidModel("MIRMIR_FP8_MODEL is not configured".into())
    })?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let bindings = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
    let layer = bindings.dense_decoder_layer(0)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let token = Array::from_u32(&[785], &[1])?;
    let input = tensors.get("model.embed_tokens.weight")?.take(&token, 0, &stream)?;
    let norm = crate::engine::NormWeight::load_name(&tensors, &layer.input_norm.source)?;
    let normalized = norm.apply(&input, 1.0e-6, &stream)?;
    let value = BoundLinear::load(&tensors, layer.attention.value, &stream)?
        .forward(&normalized, &stream)?
        .to_vec_f32(&stream)?;
    let mut attention = Vec::with_capacity(decoder.hidden_size);
    for query_head in 0..decoder.num_attention_heads {
        let kv_head = query_head * decoder.num_key_value_heads / decoder.num_attention_heads;
        attention.extend_from_slice(&value[kv_head * 64..(kv_head + 1) * 64]);
    }
    let attention = Array::from_f32(&attention, &[1, i32::try_from(decoder.hidden_size)?])?
        .astype(crate::engine::Dtype::Bfloat16, &stream)?;
    let output = BoundLinear::load(&tensors, layer.attention.output, &stream)?
        .forward(&attention, &stream)?;
    assert_values(
        &output,
        &[-0.017_456_055, 0.003_906_25, 0.011_962_891, 0.011_291_504],
        "attention output",
        &stream,
    )?;
    let residual = input.add(&output, &stream)?;
    assert_values(
        &residual,
        &[-0.049_804_688, 0.002_380_371, 0.024_047_852, -0.004_821_777_3],
        "attention residual",
        &stream,
    )?;
    let norm = crate::engine::NormWeight::load_name(&tensors, &layer.post_attention_norm.source)?;
    let normalized = norm.apply(&residual, 1.0e-6, &stream)?;
    assert_values(
        &normalized,
        &[-1.789_062_5, 0.091_308_594, 0.816_406_25, -0.194_335_94],
        "post-attention norm",
        &stream,
    )?;
    let gate = BoundLinear::load(&tensors, layer.gate, &stream)?.forward(&normalized, &stream)?;
    let up = BoundLinear::load(&tensors, layer.up, &stream)?.forward(&normalized, &stream)?;
    assert_values(
        &gate,
        &[-0.304_687_5, -1.234_375, 0.341_796_88, -0.060_058_594],
        "gate projection",
        &stream,
    )?;
    let activated = gate.silu_mul(&up, &stream)?;
    assert_values(
        &activated,
        &[-0.045_654_297, 0.000_774_383_54, -0.101_562_5, -0.006_134_033],
        "SwiGLU activation",
        &stream,
    )?;
    let down = BoundLinear::load(&tensors, layer.down, &stream)?.forward(&activated, &stream)?;
    assert_values(
        &down,
        &[-0.279_296_88, -0.190_429_69, -0.089_355_47, -0.171_875],
        "down projection",
        &stream,
    )?;
    let output = residual.add(&down, &stream)?;
    assert_values(
        &output,
        &[-0.328_125, -0.188_476_56, -0.065_429_69, -0.176_757_81],
        "layer output",
        &stream,
    )
}

fn assert_values(actual: &Array, expected: &[f32], label: &str, stream: &Stream) -> Result<()> {
    let actual = actual.to_vec_f32(stream)?;
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let tolerance = 0.01;
    assert!(
        maximum <= tolerance,
        "{label} differs from vLLM by {maximum}: {:?}",
        &actual[..expected.len()]
    );
    Ok(())
}

fn binding(scale_mode: Float8ScaleMode) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: vec![2, 2],
        logical_shape: Some(vec![2, 2]),
        transforms: Vec::new(),
        storage: TensorStorage::Float8 {
            format: Float8Quantization {
                format: Float8Format::E4M3,
                scale_mode,
                scale_granularity: Float8ScaleGranularity::OutputChannel,
                scale_dtype: Some(Float8ParameterDType::F32),
                activation_scale: Float8ActivationScale::DynamicToken,
                input_scale_dtype: None,
            },
            scale: Some("weight_scale".into()),
            input_scale: None,
            bias: Some("bias".into()),
        },
    }
}

fn static_binding(dtype: Float8ParameterDType, input_scale_name: &str) -> TensorBinding {
    let mut binding = binding(Float8ScaleMode::Multiplier);
    if let TensorStorage::Float8 { format, scale, input_scale, .. } = &mut binding.storage {
        format.activation_scale = Float8ActivationScale::StaticTensor;
        format.scale_granularity = Float8ScaleGranularity::Tensor;
        format.scale_dtype = Some(dtype);
        format.input_scale_dtype = Some(dtype);
        *scale = Some(match dtype {
            Float8ParameterDType::BF16 => "weight_scale_tensor_bf16".into(),
            Float8ParameterDType::F32 => "weight_scale_tensor".into(),
        });
        *input_scale = Some(input_scale_name.into());
    }
    binding
}

fn e5m2_binding(scaled: bool) -> TensorBinding {
    let mut binding = binding(Float8ScaleMode::Multiplier);
    binding.source = "weight_e5m2".into();
    if let TensorStorage::Float8 { format, scale, .. } = &mut binding.storage {
        *format = if scaled {
            Float8Quantization {
                format: Float8Format::E5M2,
                scale_mode: Float8ScaleMode::Multiplier,
                scale_granularity: Float8ScaleGranularity::OutputChannel,
                scale_dtype: Some(Float8ParameterDType::F32),
                activation_scale: Float8ActivationScale::None,
                input_scale_dtype: None,
            }
        } else {
            *scale = None;
            Float8Quantization::unscaled(Float8Format::E5M2)
        };
    }
    binding
}

fn write_safetensors(path: &Path) -> Result<()> {
    let weight = [0x38_u8, 0x40, 0xb8, 0x30];
    let mut payload = weight.to_vec();
    for scale in [2.0_f32, 4.0] {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    for bias in [0x3f80_u16, 0xbf80] {
        payload.extend_from_slice(&bias.to_le_bytes());
    }
    payload.extend_from_slice(&0.5_f32.to_le_bytes());
    payload.extend_from_slice(&0x3f00_u16.to_le_bytes());
    for scale in [0x4000_u16, 0x4080] {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    payload.extend_from_slice(&2.0_f32.to_le_bytes());
    payload.extend_from_slice(&0x4000_u16.to_le_bytes());
    payload.extend_from_slice(&[0x3c, 0x40, 0xbc, 0x38]);
    let mut header = r#"{"weight":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]},"weight_scale":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},"bias":{"dtype":"BF16","shape":[2],"data_offsets":[12,16]},"input_scale":{"dtype":"F32","shape":[],"data_offsets":[16,20]},"input_scale_bf16":{"dtype":"BF16","shape":[1],"data_offsets":[20,22]},"weight_scale_bf16":{"dtype":"BF16","shape":[2],"data_offsets":[22,26]},"weight_scale_tensor":{"dtype":"F32","shape":[],"data_offsets":[26,30]},"weight_scale_tensor_bf16":{"dtype":"BF16","shape":[1],"data_offsets":[30,32]},"weight_e5m2":{"dtype":"F8_E5M2","shape":[2,2],"data_offsets":[32,36]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}
