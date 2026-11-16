//! GEMV bandwidth on the Qwen3-8B `Q4_K_M` projection shapes; ignored, run with
//! `cargo test -p libmir-cuda --release ggml::tests::bandwidth -- --ignored
//! --nocapture`.

use mircuda::bf16;

use super::{Fixture, reference::Blocks};
use crate::{
    Result,
    kernels::ggml::{GgmlActivations, GgmlKernelType, GgmlLinear, GgmlSpec},
};

/// The gate projection (12288 x 4096, 28-60 MB: past the L2) in every type.
const SHAPES: [(&str, GgmlKernelType, usize, usize); 10] = [
    ("gate Q4_0", GgmlKernelType::Q4_0, 12288, 4096),
    ("gate Q4_1", GgmlKernelType::Q4_1, 12288, 4096),
    ("gate Q5_0", GgmlKernelType::Q5_0, 12288, 4096),
    ("gate Q5_1", GgmlKernelType::Q5_1, 12288, 4096),
    ("gate Q8_0", GgmlKernelType::Q8_0, 12288, 4096),
    ("gate Q2_K", GgmlKernelType::Q2K, 12288, 4096),
    ("gate Q3_K", GgmlKernelType::Q3K, 12288, 4096),
    ("gate Q4_K", GgmlKernelType::Q4K, 12288, 4096),
    ("gate Q5_K", GgmlKernelType::Q5K, 12288, 4096),
    ("gate Q6_K", GgmlKernelType::Q6K, 12288, 4096),
];

#[test]
#[ignore = "timing on the local GPU"]
#[expect(clippy::print_stdout, reason = "reports measured bandwidth")]
fn gemv_bandwidth_on_qwen3_shapes() -> Result<()> {
    let fixture = Fixture::new()?;
    for tokens in [1, 2, 10] {
        for (name, kind, rows, columns) in SHAPES {
            let blocks = Blocks::random(kind, rows, columns, 7)?;
            let weight = fixture.copy(&blocks.bytes)?;
            let input = fixture.copy(&vec![bf16::ONE; tokens * columns])?;
            let mut output =
                fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, tokens * rows)?;
            let operation = GgmlLinear::compile(
                &fixture.compiler,
                GgmlSpec::new(kind, tokens, columns, rows)?,
            )?;
            let mut activations =
                GgmlActivations::new(&fixture.pool, &fixture.stream, tokens, columns)?;
            let iterations = 50_u16;
            operation.execute(&fixture.stream, &input, &mut activations, &weight, &mut output)?;
            let start = fixture.context.create_event(true)?;
            let end = fixture.context.create_event(true)?;
            start.record(&fixture.stream)?;
            for _ in 0..iterations {
                operation
                    .execute(&fixture.stream, &input, &mut activations, &weight, &mut output)?;
            }
            end.record(&fixture.stream)?;
            end.synchronize()?;
            let microseconds = start.elapsed_ms(&end)? * 1000.0 / f32::from(iterations);
            let kilobytes = f32::from(u16::try_from(blocks.bytes.len() >> 10).unwrap_or(u16::MAX));
            println!(
                "{name:>12} tokens {tokens:>2}: {microseconds:8.1} us  {:6.1} GB/s",
                kilobytes * 1.024 / microseconds
            );
        }
    }
    Ok(())
}

/// Qwen3-8B `Q4_K_M` layers whose V and down projections are `Q6_K`.
const SIX_BIT_LAYERS: [usize; 18] =
    [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35];

#[test]
#[ignore = "timing on the local GPU"]
fn every_gemv_of_one_qwen3_token() -> Result<()> {
    for tokens in [1, 2, 4, 5, 8, 10, 16] {
        every_gemv(tokens)?;
    }
    Ok(())
}

#[expect(clippy::print_stdout, reason = "reports measured time")]
fn every_gemv(tokens: usize) -> Result<()> {
    let fixture = Fixture::new()?;
    let q4 = GgmlKernelType::Q4K;
    let q6 = GgmlKernelType::Q6K;
    let mut plan = Vec::new();
    for layer in 0..36 {
        let six = if SIX_BIT_LAYERS.contains(&layer) {
            q6
        } else {
            q4
        };
        plan.extend([
            (q4, 4096, 4096),
            (q4, 1024, 4096),
            (six, 1024, 4096),
            (q4, 4096, 4096),
            (q4, 12288, 4096),
            (q4, 12288, 4096),
            (six, 4096, 12288),
        ]);
    }
    plan.push((q6, 151_936, 4096));
    let mut steps = Vec::new();
    let mut bytes = 0;
    for (kind, rows, columns) in plan {
        let spec = GgmlSpec::new(kind, tokens, columns, rows)?;
        bytes += spec.weight_bytes()?;
        let weight = fixture.pool.allocate_zeroed::<u8>(&fixture.stream, spec.weight_bytes()?)?;
        let output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, tokens * rows)?;
        steps.push((GgmlLinear::compile(&fixture.compiler, spec)?, weight, output, columns));
    }
    let input = fixture.copy(&vec![bf16::ONE; tokens * 12288])?;
    let mut activations = [4096, 12288]
        .map(|columns| GgmlActivations::new(&fixture.pool, &fixture.stream, tokens, columns));
    let [Ok(narrow), Ok(wide)] = &mut activations else {
        return Err(crate::Error::InvalidDecoderKernel("activation allocation failed"));
    };
    let start = fixture.context.create_event(true)?;
    let end = fixture.context.create_event(true)?;
    for round in 0..4 {
        start.record(&fixture.stream)?;
        for (operation, weight, output, columns) in &mut steps {
            let activations = if *columns == 4096 {
                &mut *narrow
            } else {
                &mut *wide
            };
            operation.quantize(&fixture.stream, &input, activations)?;
            operation.multiply(&fixture.stream, activations, weight, output)?;
        }
        end.record(&fixture.stream)?;
        end.synchronize()?;
        let milliseconds = start.elapsed_ms(&end)?;
        let gigabytes = f32::from(u16::try_from(bytes >> 20).unwrap_or(u16::MAX)) / 1024.0;
        println!(
            "tokens {tokens} round {round}: {milliseconds:.2} ms for {gigabytes:.2} GiB, {:.0} GB/s",
            gigabytes * 1.073_741_8 / milliseconds * 1000.0
        );
    }
    Ok(())
}
