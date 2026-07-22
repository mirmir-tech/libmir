use super::*;

#[test]
fn creates_piecewise_rope_frequencies_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let frequencies = Array::piecewise_rope_frequencies(8, 10_000.0, 8.0, 1.0, 4.0, 8192, &stream)?;
    frequencies.async_eval()?;
    stream.synchronize()?;

    let values = frequencies.to_vec_f32()?;

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
        Array::yarn_rope_frequencies(128, 1_000_000.0, 4.0, 32.0, 1.0, 32768, &stream)?;
    frequencies.async_eval()?;
    stream.synchronize()?;

    let values = frequencies.to_vec_f32()?;
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
