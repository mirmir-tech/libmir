use super::*;

#[test]
fn creates_piecewise_rope_frequencies_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let frequencies = Array::piecewise_rope_frequencies(8, 10_000.0, 8.0, 1.0, 4.0, 8192, &stream)?;
    frequencies.async_eval(&stream)?;
    stream.synchronize()?;

    let values = frequencies.to_vec_f32(&stream)?;

    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < 1.0e-6);
    assert!((values[1] - 10.0).abs() < 1.0e-5);
    assert!((values[2] - 100.0).abs() < 1.0e-4);
    assert!(values[3] > 4_000.0 && values[3] < 5_000.0);
    Ok(())
}

#[test]
fn creates_truncated_yarn_frequencies_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let frequencies =
        Array::yarn_rope_frequencies(128, 1_000_000.0, 4.0, 32.0, 1.0, 32768, true, &stream)?;
    frequencies.async_eval(&stream)?;
    stream.synchronize()?;

    let values = frequencies.to_vec_f32(&stream)?;
    let expected = yarn_reference(128, 1_000_000.0, 4.0, 32.0, 1.0, 32768.0)?;

    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() <= expected.abs().max(1.0) * 2.0e-6);
    }
    Ok(())
}

fn yarn_reference(
    dimensions: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original: f32,
) -> Result<Vec<f32>> {
    let dimensions_f32 = f32::from(u16::try_from(dimensions)?);
    let half = dimensions_f32 / 2.0;
    let low = (half * (original / (beta_fast * std::f32::consts::TAU)).ln() / base.ln())
        .floor()
        .max(0.0);
    let high = (half * (original / (beta_slow * std::f32::consts::TAU)).ln() / base.ln())
        .ceil()
        .min(dimensions_f32 - 1.0);
    (0..dimensions / 2)
        .map(|index| {
            let index = f32::from(u16::try_from(index)?);
            let frequency = base.powf(2.0 * index / dimensions_f32);
            let ramp = ((index - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            let inverse = frequency.recip();
            Ok((inverse / factor).mul_add(1.0 - mask, inverse * mask).recip())
        })
        .collect()
}

#[test]
fn creates_continuous_yarn_frequencies_for_official_gpt_oss() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let continuous =
        Array::yarn_rope_frequencies(64, 150_000.0, 32.0, 32.0, 1.0, 4096, false, &stream)?;
    let truncated =
        Array::yarn_rope_frequencies(64, 150_000.0, 32.0, 32.0, 1.0, 4096, true, &stream)?;
    // Independent f64 correction-range calculation, with no floor/ceil.
    let expected = [
        1_f64, 1.451_285_49, 2.106_229_56, 3.056_740_39, 4.436_202_96, 6.438_196_97, 9.343_661_81,
        13.560_320_8, 19.679_896_7, 31.540_073_9, 51.719_676_3, 86.266_024_1, 147.167_912,
        259.043_244, 477.602_273, 950.026_604, 2_190.657_67, 7_732.833_61, 26_103.654_4,
        37_883.854_8, 54_980.288_6, 79_792.094_8, 115_801.109, 168_060.469, 243_903.719,
        353_973.927, 513_717.223, 745_550.35, 1_082_006.4, 1_570_300.18, 2_278_953.87,
        3_307_412.67,
    ];
    let actual = continuous.to_vec_f32(&stream)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((f64::from(*actual) - expected).abs() <= expected.max(1.0) * 3.0e-6);
    }
    let rounded = truncated.to_vec_f32(&stream)?;
    assert!((actual[12] - rounded[12]).abs() > actual[12] * 0.01);
    Ok(())
}
