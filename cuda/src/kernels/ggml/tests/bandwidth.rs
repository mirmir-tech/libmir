//! GEMV bandwidth on the Qwen3-8B `Q4_K_M` projection shapes; ignored, run with
//! `cargo test -p libmir-cuda --release ggml::tests::bandwidth -- --ignored
//! --nocapture`.

use mircuda::bf16;
use models::gguf::GgmlType;

use super::{Fixture, reference::Blocks};
use crate::{
    Result,
    kernels::ggml::{GgmlActivations, GgmlKernelType, GgmlLinear, GgmlSpec},
};

/// The gate projection (12288 x 4096, 28-60 MB: past the L2) in every type.
const SHAPES: [(&str, GgmlType, usize, usize); 10] = [
    ("gate Q4_0", GgmlType::Q4_0, 12288, 4096),
    ("gate Q4_1", GgmlType::Q4_1, 12288, 4096),
    ("gate Q5_0", GgmlType::Q5_0, 12288, 4096),
    ("gate Q5_1", GgmlType::Q5_1, 12288, 4096),
    ("gate Q8_0", GgmlType::Q8_0, 12288, 4096),
    ("gate Q2_K", GgmlType::Q2_K, 12288, 4096),
    ("gate Q3_K", GgmlType::Q3_K, 12288, 4096),
    ("gate Q4_K", GgmlType::Q4_K, 12288, 4096),
    ("gate Q5_K", GgmlType::Q5_K, 12288, 4096),
    ("gate Q6_K", GgmlType::Q6_K, 12288, 4096),
];

#[test]
#[ignore = "timing on the local GPU"]
#[expect(clippy::print_stdout, reason = "reports measured bandwidth")]
fn gemv_bandwidth_on_qwen3_shapes() -> Result<()> {
    let fixture = Fixture::new()?;
    for tokens in [1, 2, 10] {
        for (name, ggml_type, rows, columns) in SHAPES {
            let kind = GgmlKernelType::new(ggml_type)?;
            let blocks = Blocks::random(kind, rows, columns, 7)?;
            let weight = fixture.copy_blocks(&blocks.bytes)?;
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
    let (q4, q6) = (GgmlType::Q4_K, GgmlType::Q6_K);
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
    for tokens in [1, 2, 4, 5, 8, 10, 16] {
        every_gemv(tokens, &plan)?;
    }
    Ok(())
}

/// Gemma 3 12B `Q4_K_M` layers whose V and down projections are `Q6_K`.
const GEMMA_SIX_BIT_LAYERS: [usize; 24] =
    [0, 1, 2, 3, 4, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 42, 43, 44, 45, 46, 47];

#[test]
#[ignore = "timing on the local GPU"]
fn every_gemv_of_one_gemma3_token() -> Result<()> {
    let (q4, q6) = (GgmlType::Q4_K, GgmlType::Q6_K);
    let mut plan = Vec::new();
    for layer in 0..48 {
        let six = if GEMMA_SIX_BIT_LAYERS.contains(&layer) {
            q6
        } else {
            q4
        };
        plan.extend([
            (q4, 4096, 3840),
            (q4, 2048, 3840),
            (six, 2048, 3840),
            (q4, 3840, 4096),
            (q4, 15360, 3840),
            (q4, 15360, 3840),
            (six, 3840, 15360),
        ]);
    }
    plan.push((q6, 262_144, 3840));
    for tokens in [1, 2] {
        every_gemv(tokens, &plan)?;
    }
    Ok(())
}

#[expect(clippy::print_stdout, reason = "reports measured time")]
fn every_gemv(tokens: usize, plan: &[(GgmlType, usize, usize)]) -> Result<()> {
    let fixture = Fixture::new()?;
    let mut steps = Vec::new();
    let mut bytes = 0;
    for &(ggml_type, rows, columns) in plan {
        let spec = GgmlSpec::new(GgmlKernelType::new(ggml_type)?, tokens, columns, rows)?;
        bytes += spec.weight_bytes()?;
        let weight = fixture.pool.allocate_zeroed::<u8>(&fixture.stream, spec.weight_bytes()?)?;
        let output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, tokens * rows)?;
        steps.push((GgmlLinear::compile(&fixture.compiler, spec)?, weight, output, columns));
    }
    let widest = plan.iter().map(|&(_, _, columns)| columns).max().unwrap_or(0);
    let input = fixture.copy(&vec![bf16::ONE; tokens * widest])?;
    let mut activations = std::collections::BTreeMap::new();
    for &(_, _, columns) in plan {
        if let std::collections::btree_map::Entry::Vacant(entry) = activations.entry(columns) {
            entry.insert(GgmlActivations::new(&fixture.pool, &fixture.stream, tokens, columns)?);
        }
    }
    let start = fixture.context.create_event(true)?;
    let end = fixture.context.create_event(true)?;
    for round in 0..4 {
        start.record(&fixture.stream)?;
        for (operation, weight, output, columns) in &mut steps {
            let activations = activations
                .get_mut(columns)
                .ok_or(crate::Error::InvalidDecoderKernel("activation width missing"))?;
            operation.quantize(&fixture.stream, &input, activations)?;
            operation.multiply(&fixture.stream, activations, weight, output)?;
        }
        end.record(&fixture.stream)?;
        end.synchronize()?;
        let milliseconds = start.elapsed_ms(&end)?;
        if round == 3 {
            by_shape(&fixture, &mut steps, &input, &mut activations)?;
        }
        let gigabytes = f32::from(u16::try_from(bytes >> 20).unwrap_or(u16::MAX)) / 1024.0;
        println!(
            "tokens {tokens} round {round}: {milliseconds:.2} ms for {gigabytes:.2} GiB, {:.0} GB/s",
            gigabytes * 1.073_741_8 / milliseconds * 1000.0
        );
    }
    Ok(())
}

type Step = (GgmlLinear, mircuda::DeviceBuffer<u8>, mircuda::DeviceBuffer<bf16>, usize);

/// Each step timed alone (events between steps), summed per type and shape.
#[expect(clippy::print_stdout, reason = "reports measured time")]
fn by_shape(
    fixture: &Fixture,
    steps: &mut [Step],
    input: &mircuda::DeviceBuffer<bf16>,
    activations: &mut std::collections::BTreeMap<usize, GgmlActivations>,
) -> Result<()> {
    let mut sums = std::collections::BTreeMap::<String, (f32, f32, usize)>::new();
    for (operation, weight, output, columns) in steps {
        let activations = activations
            .get_mut(columns)
            .ok_or(crate::Error::InvalidDecoderKernel("activation width missing"))?;
        operation.quantize(&fixture.stream, input, activations)?;
        let start = fixture.context.create_event(true)?;
        let end = fixture.context.create_event(true)?;
        start.record(&fixture.stream)?;
        operation.multiply(&fixture.stream, activations, weight, output)?;
        end.record(&fixture.stream)?;
        end.synchronize()?;
        let spec = operation.spec();
        let key = format!("{:?} {}x{}", spec.kind, spec.output_features, spec.input_features);
        let entry = sums.entry(key).or_insert((0.0, 0.0, 0));
        entry.0 += start.elapsed_ms(&end)?;
        #[expect(clippy::cast_precision_loss, reason = "a report")]
        let bytes = spec.weight_bytes()? as f32;
        entry.1 += bytes;
        entry.2 += 1;
    }
    for (key, (milliseconds, bytes, count)) in sums {
        println!(
            "  {key:>22} x{count:<3} {milliseconds:7.2} ms {:6.1} GB/s",
            bytes / milliseconds / 1e6
        );
    }
    Ok(())
}
