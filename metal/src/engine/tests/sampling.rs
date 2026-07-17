use super::*;
use crate::engine::{DeviceSampling, sample, sample_u32};

#[test]
fn samples_top_p_and_top_k_without_copying_logits() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = Array::from_f32(&[5.0, 4.0, 1.0], &[1, 1, 3])?;
    let base = DeviceSampling {
        vocab_size: 3,
        top_k: 2,
        top_p: 0.9,
        temperature: 1.0,
        draw: 0.0,
    };

    assert_eq!(sample(&logits, base, &stream)?.shape()?, vec![1, 1]);
    assert_eq!(sample_u32(&logits, base, &stream)?, 0);
    assert_eq!(sample_u32(&logits, DeviceSampling { draw: 0.99, ..base }, &stream)?, 1);
    assert_eq!(
        sample_u32(&logits, DeviceSampling { top_p: 0.7, draw: 0.99, ..base }, &stream)?,
        0
    );
    Ok(())
}
