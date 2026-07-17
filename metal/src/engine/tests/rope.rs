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
