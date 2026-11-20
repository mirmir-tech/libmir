use super::{DeviceSampling, sample, sample_rows};
use crate::engine::{Array, Result, Stream};

const ROWS: usize = 6;
const VOCAB: usize = 97;

#[test]
fn shared_parameter_rows_sample_like_individual_rows() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let data = (0..ROWS * VOCAB)
        .map(|i| f32::from(u16::try_from((i * 37 + i / VOCAB * 11) % 101).unwrap_or(0)) / 9.0)
        .collect::<Vec<_>>();
    let logits = Array::from_f32(&data, &[i32::try_from(ROWS)?, 1, i32::try_from(VOCAB)?])?;
    let draws = [0.0, 0.13, 0.41, 0.5, 0.77, 0.999];
    for (top_k, top_p) in [(20, 0.95), (0, 1.0), (5, 1.0)] {
        let parameters = DeviceSampling {
            vocab_size: VOCAB,
            top_k,
            top_p,
            temperature: 0.6,
            draw: 0.0,
        };
        let batched = sample_rows(&logits, parameters, &draws, &stream)?.to_vec_u32(&stream)?;
        let single = (0..ROWS)
            .map(|row| {
                let row_logits = logits.slice(&[row, 0, 0], &[row + 1, 1, VOCAB], &stream)?;
                let parameters = DeviceSampling { draw: draws[row], ..parameters };
                Ok(sample(&row_logits, parameters, &stream)?.to_vec_u32(&stream)?[0])
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(batched, single, "top_k {top_k} top_p {top_p}");
    }
    Ok(())
}
