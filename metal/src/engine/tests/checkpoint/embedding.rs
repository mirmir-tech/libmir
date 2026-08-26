#![allow(clippy::print_stdout)]

use std::{env, hint::black_box, time::Instant};

use super::*;

#[test]
#[ignore = "benchmark; set MIRMIR_BENCH_MODEL or MODEL"]
fn bench_real_gemma_quantized_embedding() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let ids = Array::from_u32(&[1], &[1, 1])?;
    let iterations = env_usize("MIRMIR_BENCH_ITERS", 100)?;
    let warmup = env_usize("MIRMIR_BENCH_WARMUP", 20)?;

    for _ in 0..warmup {
        let output = embedding.lookup(&ids, &stream)?;
        output.async_eval(&stream)?;
        stream.synchronize()?;
        black_box(output);
    }
    let started = Instant::now();
    for _ in 0..iterations {
        let output = embedding.lookup(&ids, &stream)?;
        output.async_eval(&stream)?;
        stream.synchronize()?;
        black_box(output);
    }
    let iterations = f64::from(u32::try_from(iterations)?);
    let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / iterations;
    println!("mirmir_embedding_bench ms={milliseconds:.6}");
    Ok(())
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(_) => Ok(default),
    }
}
