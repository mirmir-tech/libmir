use super::*;

const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub(super) fn decode(word: u32, lane: usize, exponent: u32) -> f64 {
    let code = (word >> (lane * 4)) & 15;
    let sign = if code & 8 == 0 {
        1.0
    } else {
        -1.0
    };
    sign * MAGNITUDES[(code & 7) as usize] * (f64::from(exponent) - 127.0).exp2()
}

pub(super) fn dot_samples(
    projection: &MxFp4Linear,
    input: &Array,
    stream: &Stream,
) -> Result<Vec<f64>> {
    Ok(samples(projection, input, stream, 1)?.full)
}

pub(super) struct Samples {
    pub full: Vec<f64>,
    pub bf16_partials: Vec<f64>,
    pub fp32_partials: Vec<f64>,
    pub bf16_tree: Vec<f64>,
}

pub(super) fn samples(
    projection: &MxFp4Linear,
    input: &Array,
    stream: &Stream,
    partitions: usize,
) -> Result<Samples> {
    let weights = projection.weight.to_vec_u32(stream)?;
    let scales = projection.scales.astype(Dtype::Uint32, stream)?.to_vec_u32(stream)?;
    assert!(scales.iter().all(|&scale| scale < 255), "reserved E8M0 NaN scale");
    let input = input.to_vec_f32(stream)?;
    let k = projection.input_features;
    let columns = columns(projection.output_features);
    let decoded = columns
        .iter()
        .map(|&column| {
            (0..k)
                .map(|i| {
                    decode(
                        weights[column * (k / 8) + i / 8],
                        i % 8,
                        scales[column * (k / 32) + i / 32],
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(partitions > 0 && k.is_multiple_of(partitions));
    let mut result = Samples {
        full: Vec::new(),
        bf16_partials: Vec::new(),
        fp32_partials: Vec::new(),
        bf16_tree: Vec::new(),
    };
    for row in input.chunks_exact(k) {
        for column in &decoded {
            let products =
                row.iter().zip(column).map(|(x, w)| f64::from(*x) * w).collect::<Vec<_>>();
            result.full.push(products.iter().sum());
            let partials = products
                .chunks_exact(k / partitions)
                .map(|chunk| chunk.iter().sum::<f64>())
                .collect::<Vec<_>>();
            result
                .bf16_partials
                .push(nearest_bf16(partials.iter().map(|&v| nearest_bf16(v)).sum()));
            result
                .fp32_partials
                .push(nearest_bf16(partials.iter().map(|&v| f64::from(round_fp32(v))).sum()));
            let lanes = partials.len().min(8);
            let totals = (0..lanes).map(|lane| {
                partials
                    .iter()
                    .skip(lane)
                    .step_by(lanes)
                    .fold(0.0, |sum, &v| nearest_bf16(sum + nearest_bf16(v)))
            });
            result.bf16_tree.push(totals.fold(0.0, |sum, v| nearest_bf16(sum + v)));
        }
    }
    Ok(result)
}

pub(super) fn columns(width: usize) -> Vec<usize> {
    (0..16).map(|i| i * (width - 1) / 15).collect()
}

#[expect(clippy::cast_possible_truncation, reason = "explicit FP32 partition-rounding control")]
fn round_fp32(value: f64) -> f32 {
    value as f32
}

pub(super) fn nearest_bf16(value: f64) -> f64 {
    let rounded = round_fp32(value);
    let bits = rounded.to_bits() >> 16;
    // Inspect adjacent BF16 values against the original f64 sum, avoiding
    // double rounding through f32 at a BF16 midpoint.
    [bits.saturating_sub(1), bits, bits + 1]
        .into_iter()
        .map(|bits| (bits, f64::from(f32::from_bits(bits << 16))))
        .filter(|(_, candidate)| candidate.is_finite())
        .min_by(|(a_bits, a), (b_bits, b)| {
            (a - value)
                .abs()
                .total_cmp(&(b - value).abs())
                .then_with(|| (a_bits & 1).cmp(&(b_bits & 1)))
        })
        .map_or(0.0, |(_, value)| value)
}

#[test]
#[expect(clippy::float_cmp, reason = "exact FP4 values and nearest-even BF16 boundary cases")]
fn reference_decodes_signed_fp4_exponents_and_rounds_ties() {
    let word = 0xfedc_ba98;
    for (lane, magnitude) in MAGNITUDES.into_iter().enumerate() {
        assert_eq!(decode(word, lane, 127), -magnitude);
        assert_eq!(decode(word, lane, 125), -magnitude / 4.0);
        assert_eq!(decode(0x7654_3210, lane, 128), magnitude * 2.0);
    }
    assert_eq!(decode(2, 0, 0), 2.0f64.powi(-127));
    assert_eq!(nearest_bf16(1.0 + 1.0 / 256.0), 1.0);
    assert_eq!(nearest_bf16(1.0 + 1.0 / 256.0 + 1e-10), 1.0 + 1.0 / 128.0);
    assert_eq!(nearest_bf16(-1.0 - 1.0 / 256.0), -1.0);
    assert_eq!(nearest_bf16(-1.0 - 1.0 / 256.0 - 1e-10), -1.0 - 1.0 / 128.0);
}

#[test]
fn fixed_reference_rounds_product_before_output_bias() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut values = vec![0.0; 32];
    values[0] = 1.0;
    values[1] = 1.0 / 256.0;
    let input = Array::from_f32(&values, &[1, 32])?.astype(Dtype::Bfloat16, &stream)?;
    let weight = Array::from_u32(&[0x2222_2222; 128], &[32, 4])?;
    let scales = Array::from_u32(&[127; 32], &[32, 1])?.astype(Dtype::Uint8, &stream)?;
    let bias = Array::from_f32(&[1.0 / 256.0; 32], &[32])?.astype(Dtype::Bfloat16, &stream)?;
    let native = Array::from_native(stream.native().graph().mxfp4_matmul(
        input.native(),
        mirtal::MxFp4 {
            weight: weight.native(),
            scales: scales.native(),
        },
        true,
    )?)?
    .add(&bias, &stream)?;
    assert_eq!(native.to_vec_f32(&stream)?, vec![1.0; 32]);
    let fixed =
        stream
            .kernels()
            .mxfp4_linear([&input, &weight, &scales, &bias], 32, 32, &stream)?;
    assert_eq!(
        fixed.to_vec_f32(&stream)?,
        native.to_vec_f32(&stream)?,
        "output bias follows BF16 projection rounding"
    );
    Ok(())
}
