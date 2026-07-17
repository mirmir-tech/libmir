use std::{env, hint::black_box, io::Write, time::Instant};

use super::*;

#[test]
#[ignore = "benchmark; set MIRMIR_BENCH_MODEL or MODEL"]
fn bench_real_gemma_output_head() -> Result<()> {
    let (tensors, stream) = load_model()?;
    let embedding = QuantizedEmbedding::load(&tensors, "language_model.model.embed_tokens", 64)?;
    let norm = tensors.get("language_model.model.norm.weight")?;
    let hidden =
        Array::from_f32(&vec![0.25; 2_816], &[1, 1, 2_816])?.astype_like(&norm, &stream)?;
    let normalized = hidden.rms_norm(&norm, 1.0e-6, &stream)?;
    let iterations = env_usize("MIRMIR_BENCH_ITERS", 40)?;
    let warmup = env_usize("MIRMIR_BENCH_WARMUP", 10)?;

    let norm_ms = measure(iterations, warmup, &stream, || hidden.rms_norm(&norm, 1.0e-6, &stream))?;
    let projection_ms =
        measure(iterations, warmup, &stream, || embedding.project(&normalized, &stream))?;
    let logits = embedding.project(&normalized, &stream)?;
    let argmax_ms = measure(iterations, warmup, &stream, || logits.argmax(&stream))?;
    let projection_argmax_ms = measure(iterations, warmup, &stream, || {
        embedding.project(&normalized, &stream)?.argmax(&stream)
    })?;

    let mut report = std::io::stderr().lock();
    writeln!(
        report,
        "mirmir_output_head_bench iters={iterations} warmup={warmup} final_norm_ms={norm_ms:.4} projection_ms={projection_ms:.4} argmax_ms={argmax_ms:.4} projection_argmax_ms={projection_argmax_ms:.4}"
    )?;
    Ok(())
}

fn measure(
    iterations: usize,
    warmup: usize,
    stream: &Stream,
    mut run: impl FnMut() -> Result<Array>,
) -> Result<f64> {
    for _ in 0..warmup {
        let output = run()?;
        output.async_eval()?;
        stream.synchronize()?;
        black_box(output);
    }
    let started = Instant::now();
    for _ in 0..iterations {
        let output = run()?;
        output.async_eval()?;
        stream.synchronize()?;
        black_box(output);
    }
    let iterations = f64::from(u32::try_from(iterations)?);
    Ok(started.elapsed().as_secs_f64() * 1_000.0 / iterations)
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(_) => Ok(default),
    }
}
