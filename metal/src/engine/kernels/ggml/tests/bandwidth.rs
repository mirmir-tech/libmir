//! Ignored timing of the GGML GEMV at Qwen3-8B projection shapes:
//! `cargo test -p libmir-metal --release --lib ggml::tests::bandwidth --
//! --ignored --nocapture`.
#![allow(clippy::print_stdout)]

use std::time::Instant;

use models::gguf::GgmlType;

use super::{blocks, weight};
use crate::engine::{Array, Dtype, Result, Stream, kernels::ggml::GgmlMatrix};

#[test]
#[ignore = "timing on the local GPU"]
fn gemv_bandwidth() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let shapes: Vec<(GgmlType, usize, usize)> =
        match std::env::var("GGML_BANDWIDTH_SHAPES").as_deref() {
            Ok("qwen3-8b") => vec![
                (GgmlType::Q4_K, 4096, 4096),
                (GgmlType::Q4_K, 4096, 1024),
                (GgmlType::Q6_K, 4096, 1024),
                (GgmlType::Q4_K, 4096, 12288),
                (GgmlType::Q6_K, 12288, 4096),
                (GgmlType::Q6_K, 4096, 151_936),
            ],
            _ => [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::Q5_K]
                .into_iter()
                .flat_map(|ggml_type| [(ggml_type, 4096, 12288), (ggml_type, 12288, 4096)])
                .collect(),
        };
    for (ggml_type, columns, rows) in shapes {
        {
            // Copies well beyond the system cache, cycled so each launch
            // streams its weights from memory as a model's layers do.
            let data = blocks(ggml_type, rows, columns, 5)?;
            let bytes = data.len();
            let copies = (256 << 20) / bytes + 1;
            let weights =
                (0..copies).map(|_| weight(&data, rows, &stream)).collect::<Result<Vec<_>>>()?;
            stream.eval_many(&weights.iter().collect::<Vec<_>>())?;
            let matrix = GgmlMatrix::new(ggml_type, columns, &weights[0])?;
            for tokens in [1, 2, 8, 16, 32, 64] {
                let input = Array::from_f32(
                    &vec![0.5; tokens * columns],
                    &[i32::try_from(tokens)?, i32::try_from(columns)?],
                )?
                .astype(Dtype::Bfloat16, &stream)?;
                let run = |count: usize| -> Result<f64> {
                    let start = Instant::now();
                    let mut outputs = Vec::with_capacity(count);
                    for index in 0..count {
                        let weight = &weights[index % copies];
                        outputs.push(stream.kernels().ggml_gemv(&stream, &input, weight, matrix)?);
                    }
                    stream.eval_many(&outputs.iter().collect::<Vec<_>>())?;
                    stream.synchronize()?;
                    Ok(start.elapsed().as_secs_f64() / f64::from(u32::try_from(count)?))
                };
                run(16)?;
                let enqueue = Instant::now();
                let pending = (0..256)
                    .map(|index| {
                        stream.kernels().ggml_gemv(
                            &stream,
                            &input,
                            &weights[index % copies],
                            matrix,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let enqueue = enqueue.elapsed().as_secs_f64() / 256.0;
                drop(pending);
                let mut seconds = f64::INFINITY;
                for _ in 0..5 {
                    seconds = seconds.min(run(4 * copies)?);
                }
                println!(
                    "{ggml_type} {columns}->{rows} t{tokens}: {:.1} us, {:.0} GB/s, enqueue {:.1} us",
                    seconds * 1.0e6,
                    f64::from(u32::try_from(bytes)?) / seconds / 1.0e9,
                    enqueue * 1.0e6
                );
            }
        }
    }
    Ok(())
}
