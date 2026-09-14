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

#[test]
fn samples_a_greedy_batch_with_one_argmax() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = Array::from_f32(&[5.0, 4.0, 1.0, 2.0, 3.0, 4.0], &[2, 1, 3])?;
    let tokens = logits.argmax(&stream)?;
    tokens.async_eval(&stream)?;

    assert_eq!(tokens.shape()?, vec![2, 1]);
    assert_eq!(tokens.to_vec_u32(&stream)?, vec![0, 2]);
    Ok(())
}

#[test]
fn full_vocabulary_sampling_preserves_mass_and_excludes_padding() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = Array::from_f32(&[-1000.0, 0.0, 0.0, 100.0], &[1, 1, 4])?
        .astype(Dtype::Bfloat16, &stream)?;
    let base = DeviceSampling {
        vocab_size: 3,
        top_k: 0,
        top_p: 1.0,
        temperature: 0.7,
        draw: 0.0,
    };
    for (draw, expected) in [(0.0, 1), (0.49, 1), (0.51, 2), (0.999_999_94, 2)] {
        assert_eq!(sample_u32(&logits, DeviceSampling { draw, ..base }, &stream)?, expected);
    }
    assert!(sample(&logits, DeviceSampling { top_p: 0.9, ..base }, &stream).is_err());
    Ok(())
}

#[test]
fn full_vocabulary_sampling_uses_fp32_cumulative_mass() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits =
        Array::from_f32(&vec![0.0; 4096], &[2, 1, 2048])?.astype(Dtype::Bfloat16, &stream)?;
    for (draw, expected) in [(0.1, 204), (0.5, 1024), (0.9, 1843), (0.999_999_94, 2047)] {
        let tokens = sample(
            &logits,
            DeviceSampling {
                vocab_size: 2048,
                top_k: 0,
                top_p: 1.0,
                temperature: 1.0,
                draw,
            },
            &stream,
        )?;
        assert_eq!(tokens.to_vec_u32(&stream)?, vec![expected; 2]);
    }
    Ok(())
}
