use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use libmir_cuda::kernels::{
    DirectFp8Activation, DirectFp8Linear, DirectFp8Scale, DirectFp8Scales, DirectFp8Spec,
};
use mircuda::bf16;
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

use super::{copy_device, read, resources};

#[test]
#[ignore = "requires MIRMIR_FP8_MODEL"]
fn real_qwen2_projection_matches_direct_dequantization() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::var("MIRMIR_FP8_MODEL")?;
    let catalog = TensorCatalog::from_layout(&ModelLayout::inspect(Path::new(&root))?)?;
    let prefix = "model.layers.0.self_attn.q_proj";
    let weight = payload(required(&catalog, &format!("{prefix}.weight"))?)?;
    let scales = bf16_payload(required(&catalog, &format!("{prefix}.weight_scale"))?)?;
    let bias = bf16_payload(required(&catalog, &format!("{prefix}.bias"))?)?;
    let rows = bias.len();
    let columns = weight.len() / rows;
    let input = (0..columns)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 29)?) / 32.0 - 0.4375)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let (context, stream, pool, compiler) = resources()?;
    let input_device = copy_device(&context, &stream, &pool, &input)?;
    let weight_device = copy_device(&context, &stream, &pool, &weight)?;
    let scale_device = copy_device(&context, &stream, &pool, &scales)?;
    let bias_device = copy_device(&context, &stream, &pool, &bias)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, rows)?;
    DirectFp8Linear::compile(
        &compiler,
        DirectFp8Spec::new(
            1,
            columns,
            rows,
            DirectFp8Scale::OutputChannel,
            false,
            DirectFp8Activation::DynamicE4M3Token,
        )?,
    )?
    .execute_bf16_scales(
        &stream,
        &input_device,
        &weight_device,
        DirectFp8Scales {
            weight: &scale_device,
            activation: &scale_device,
        },
        Some(&bias_device),
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = reference(&input, &weight, &scales, &bias);
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.031_25, "maximum real-checkpoint projection error: {error}");
    Ok(())
}

fn reference(input: &[bf16], weight: &[u8], scales: &[bf16], bias: &[bf16]) -> Vec<bf16> {
    let maximum = input.iter().map(|value| value.to_f32().abs()).fold(0.0_f32, f32::max);
    let activation_scale = (maximum / 448.0).max(1.0 / (448.0 * 512.0));
    let input = input
        .iter()
        .map(|value| nearest_e4m3(value.to_f32() / activation_scale) * activation_scale)
        .collect::<Vec<_>>();
    weight
        .chunks_exact(input.len())
        .zip(scales)
        .zip(bias)
        .map(|((row, scale), bias)| {
            let sum = input
                .iter()
                .zip(row)
                .map(|(input, weight)| input * e4m3(*weight) * scale.to_f32())
                .sum::<f32>();
            bf16::from_f32(sum + bias.to_f32())
        })
        .collect()
}

fn nearest_e4m3(value: f32) -> f32 {
    (0_u8..=u8::MAX)
        .filter(|bits| bits & 0x7f != 0x7f)
        .map(e4m3)
        .min_by(|left, right| {
            (left - value)
                .abs()
                .total_cmp(&(right - value).abs())
                .then_with(|| left.abs().total_cmp(&right.abs()))
        })
        .unwrap_or(0.0)
}

fn e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 {
        1.0
    } else {
        -1.0
    };
    let exponent = i32::from((bits >> 3) & 0x0f);
    let mantissa = f32::from(bits & 0x07);
    let magnitude = if exponent == 0 {
        mantissa * 2.0_f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(exponent - 7)
    };
    sign * magnitude
}

fn bf16_payload(info: &TensorInfo) -> Result<Vec<bf16>, Box<dyn std::error::Error>> {
    Ok(payload(info)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)))
        .collect())
}

fn payload(info: &TensorInfo) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(info.payload_start()?))?;
    let mut bytes = vec![0; info.payload_bytes()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn required<'a>(
    catalog: &'a TensorCatalog,
    name: &str,
) -> Result<&'a TensorInfo, Box<dyn std::error::Error>> {
    catalog.get(name).ok_or_else(|| format!("missing {name}").into())
}
