use libmir_cuda::{
    Result,
    kernels::{
        DirectFp8Activation, DirectFp8Linear, DirectFp8Scale, DirectFp8Scales, DirectFp8Spec,
    },
};
use mircuda::bf16;

use super::{INPUT, OUTPUT, TOKENS, copy_device, read, resources, weight_bytes, weight_values};

#[test]
fn static_e4m3_tensor_scale_matches_independent_fake_quantization() -> Result<()> {
    let activation_scale = 0.25_f32;
    let weight_scale_value = 0.5_f32;
    let weight_scale = [weight_scale_value];
    let input_values = input_values();
    let (context, stream, pool, compiler) = resources()?;
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &weight_bytes())?;
    let weight_scale = copy_device(&context, &stream, &pool, &weight_scale)?;
    let input_scale = copy_device(&context, &stream, &pool, &[activation_scale])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, TOKENS * OUTPUT)?;
    let spec = DirectFp8Spec::new(
        TOKENS,
        INPUT,
        OUTPUT,
        DirectFp8Scale::Tensor,
        false,
        DirectFp8Activation::StaticE4M3Tensor,
    )?;
    DirectFp8Linear::compile(&compiler, spec)?.execute(
        &stream,
        &input,
        &weight,
        DirectFp8Scales {
            weight: &weight_scale,
            activation: &input_scale,
        },
        None,
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    assert_eq!(actual, reference(&input_values, activation_scale, weight_scale_value));
    Ok(())
}

#[test]
fn static_e4m3_bf16_scales_match_independent_fake_quantization() -> Result<()> {
    let activation_scale = bf16::from_f32(0.25);
    let weight_scale_value = bf16::from_f32(0.5);
    let weight_scale = [weight_scale_value];
    let input_values = input_values();
    let (context, stream, pool, compiler) = resources()?;
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &weight_bytes())?;
    let weight_scale = copy_device(&context, &stream, &pool, &weight_scale)?;
    let input_scale = copy_device(&context, &stream, &pool, &[activation_scale])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, TOKENS * OUTPUT)?;
    let spec = DirectFp8Spec::new(
        TOKENS,
        INPUT,
        OUTPUT,
        DirectFp8Scale::Tensor,
        false,
        DirectFp8Activation::StaticE4M3Tensor,
    )?;
    DirectFp8Linear::compile(&compiler, spec)?.execute_bf16_scales(
        &stream,
        &input,
        &weight,
        DirectFp8Scales {
            weight: &weight_scale,
            activation: &input_scale,
        },
        None,
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    assert_eq!(
        actual,
        reference(&input_values, activation_scale.to_f32(), weight_scale_value.to_f32())
    );
    Ok(())
}

fn input_values() -> Vec<bf16> {
    [
        0.53, -0.47, 1.13, -1.07, 0.19, -0.21, 2.26, -2.14, 0.31, 0.61, -0.91, 1.21, -1.51, 1.81,
        -2.11, 2.41,
    ]
    .into_iter()
    .map(bf16::from_f32)
    .collect()
}

fn reference(input: &[bf16], activation_scale: f32, weight_scale: f32) -> Vec<bf16> {
    let input = input
        .iter()
        .map(|value| nearest_e4m3(value.to_f32() / activation_scale) * activation_scale)
        .collect::<Vec<_>>();
    let weights = weight_values();
    let mut output = Vec::with_capacity(TOKENS * OUTPUT);
    for token in input.as_chunks::<INPUT>().0 {
        for row in weights.as_chunks::<INPUT>().0 {
            let sum = token
                .iter()
                .zip(row)
                .map(|(input, weight)| input * weight * weight_scale)
                .sum::<f32>();
            output.push(bf16::from_f32(sum));
        }
    }
    output
}

fn nearest_e4m3(value: f32) -> f32 {
    (0_u8..=u8::MAX)
        .filter(|bits| bits & 0x7f != 0x7f)
        .map(e4m3)
        .min_by(|left, right| (left - value).abs().total_cmp(&(right - value).abs()))
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
