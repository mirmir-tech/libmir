//! Ignored timing of one `Qwen3-8B` `Q4_K_M` decode step's GGML projections,
//! chained layer by layer as the model runs them:
//! `cargo test -p libmir-metal --release --lib ggml::tests::chain -- --ignored
//! --nocapture`.
#![allow(clippy::print_stdout)]

use std::time::Instant;

use models::gguf::GgmlType;

use super::{blocks, weight};
use crate::engine::{Array, Dtype, Result, Stream, kernels::ggml::GgmlMatrix};

struct Layer {
    projections: Vec<(Array, GgmlMatrix)>,
}

fn projection(
    ggml_type: GgmlType,
    columns: usize,
    rows: usize,
    seed: u64,
    stream: &Stream,
) -> Result<(Array, GgmlMatrix)> {
    let data = blocks(ggml_type, rows, columns, seed)?;
    let array = weight(&data, rows, stream)?;
    let matrix = GgmlMatrix::new(ggml_type, columns, &array)?;
    Ok((array, matrix))
}

#[test]
#[ignore = "timing on the local GPU"]
fn chain() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let layers = (0..8)
        .map(|layer| {
            let seed = 100 * layer;
            Ok(Layer {
                projections: vec![
                    projection(GgmlType::Q4_K, 4096, 4096, seed + 1, &stream)?,
                    projection(GgmlType::Q4_K, 4096, 1024, seed + 2, &stream)?,
                    projection(GgmlType::Q6_K, 4096, 1024, seed + 3, &stream)?,
                    projection(GgmlType::Q4_K, 4096, 4096, seed + 4, &stream)?,
                    projection(GgmlType::Q4_K, 4096, 12288, seed + 5, &stream)?,
                    projection(GgmlType::Q4_K, 4096, 12288, seed + 6, &stream)?,
                    projection(GgmlType::Q6_K, 12288, 4096, seed + 7, &stream)?,
                ],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for layer in &layers {
        stream.eval_many(&layer.projections.iter().map(|(array, _)| array).collect::<Vec<_>>())?;
    }
    for tokens in [1_usize, 2, 4, 8] {
        let step = || -> Result<f64> {
            let mut hidden =
                Array::from_f32(&vec![0.01; tokens * 4096], &[i32::try_from(tokens)?, 1, 4096])?
                    .astype(Dtype::Bfloat16, &stream)?;
            let started = Instant::now();
            let mut side = Vec::new();
            for index in 0..36 {
                let layer = &layers[index % layers.len()];
                let run = |input: &Array, at: usize| {
                    let (array, matrix) = &layer.projections[at];
                    stream.kernels().ggml_gemv(&stream, input, array, *matrix)
                };
                let query = run(&hidden, 0)?;
                side.push(run(&hidden, 1)?);
                side.push(run(&hidden, 2)?);
                let attended = run(&query, 3)?;
                let gate = run(&attended, 4)?;
                let up = run(&attended, 5)?;
                hidden = run(&gate.silu_mul(&up, &stream)?, 6)?;
            }
            side.push(hidden);
            stream.eval_many(&side.iter().collect::<Vec<_>>())?;
            stream.synchronize()?;
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        };
        step()?;
        let best = (0..5)
            .map(|_| step())
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        println!("tokens {tokens}: {best:.2} ms per 36-layer step");
    }
    Ok(())
}
