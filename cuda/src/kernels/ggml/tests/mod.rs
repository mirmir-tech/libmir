//! GGML kernels against a host port of llama.cpp's `dequantize_row_*`.

use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};
use models::gguf::GgmlType;

use super::{
    GgmlActivations, GgmlDequantize, GgmlEmbedding, GgmlEmbeddingOperands, GgmlKernelType,
    GgmlLinear, GgmlSpec,
};
use crate::Result;

mod bandwidth;
mod gathered;
mod mmq;
mod reference;

use reference::{Blocks, dequantize};

const ROWS: usize = 24;
const COLUMNS: usize = 512;
#[test]
fn linear_kernels_match_the_reference_dequantization() -> Result<()> {
    let fixture = Fixture::new()?;
    // Passes up to a type's `staged_tokens` take the shared-memory kernel:
    // 2048 columns hold whole 64-sub-block chunks, 512 and 3840 end in a
    // partial one, and their rows of odd block counts start misaligned.
    let shapes = |kind| [(kind, COLUMNS), (kind, 2048), (kind, 3840)];
    for (kind, columns) in GgmlKernelType::all().flat_map(shapes) {
        let blocks = Blocks::random(kind, ROWS, columns, 0x9e37_79b9)?;
        let weight = dequantize(kind, &blocks.bytes, ROWS, columns);
        let device_weight = fixture.copy(&blocks.bytes)?;
        // Passes of 1, 2, 8 and 16 tokens; 37 take three 16-token passes
        // with a ragged last one.
        for tokens in [1, 2, 5, 37] {
            let input = reference::input(tokens * columns, 0x85eb_ca6b);
            let device_input = fixture.copy(&input)?;
            let mut output =
                fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, tokens * ROWS)?;
            let spec = GgmlSpec::new(kind, tokens, columns, ROWS)?;
            let mut activations =
                GgmlActivations::new(&fixture.pool, &fixture.stream, tokens, columns)?;
            GgmlLinear::compile(&fixture.compiler, spec)?.execute(
                &fixture.stream, &device_input, &mut activations, &device_weight, &mut output,
            )?;
            let actual = fixture.read(&output)?;
            for token in 0..tokens {
                for row in 0..ROWS {
                    let (expected, magnitude) = reference::dot(
                        &input[token * columns..(token + 1) * columns],
                        &weight[row * columns..(row + 1) * columns],
                    );
                    let value = f64::from(actual[token * ROWS + row].to_f32());
                    assert!(
                        (value - expected).abs() <= 1e-2f64.mul_add(magnitude, 1e-3),
                        "{kind:?} tokens {tokens} [{token}, {row}]: {value} vs {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn embedding_rows_match_the_reference_dequantization() -> Result<()> {
    let fixture = Fixture::new()?;
    for kind in GgmlKernelType::all() {
        let blocks = Blocks::random(kind, ROWS, COLUMNS, 0xc2b2_ae35)?;
        let weight = dequantize(kind, &blocks.bytes, ROWS, COLUMNS);
        let selected = [5_u32, 0, 23, 5];
        let device_weight = fixture.copy(&blocks.bytes)?;
        let device_selected = fixture.copy(&selected)?;
        let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 3 * COLUMNS)?;
        let operation = GgmlEmbedding::compile(&fixture.compiler, kind, ROWS, COLUMNS, 2.0)?;
        operation.execute(
            &fixture.stream,
            &mut GgmlEmbeddingOperands {
                weight: &device_weight,
                selected: &device_selected,
                output: &mut output,
            },
            1,
            3,
        )?;
        let actual = fixture.read(&output)?;
        for (token, row) in [0_usize, 23, 5].into_iter().enumerate() {
            for column in 0..COLUMNS {
                let expected = 2.0 * weight[row * COLUMNS + column];
                let value = f64::from(actual[token * COLUMNS + column].to_f32());
                assert!(
                    (value - expected).abs() <= expected.abs() / 128.0 + 1e-6,
                    "{kind:?} row {row} column {column}: {value} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn dequantization_matches_the_reference() -> Result<()> {
    let fixture = Fixture::new()?;
    for kind in GgmlKernelType::all() {
        let blocks = Blocks::random(kind, ROWS, COLUMNS, 0x27d4_eb2f)?;
        let weight = dequantize(kind, &blocks.bytes, ROWS, COLUMNS);
        let device_weight = fixture.copy(&blocks.bytes)?;
        let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, ROWS * COLUMNS)?;
        GgmlDequantize::compile(&fixture.compiler, kind, ROWS, COLUMNS)?
            .execute(&fixture.stream, &device_weight, &mut output)?;
        // BF16: half a unit in the last place of an 8-bit mantissa.
        for (index, (value, expected)) in fixture.read(&output)?.into_iter().zip(weight).enumerate()
        {
            let value = f64::from(value.to_f32());
            assert!(
                (value - expected).abs() <= expected.abs() / 256.0 + 1e-6,
                "{kind:?} element {index}: {value} vs {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn rejects_partial_blocks_and_short_weights() -> Result<()> {
    assert!(GgmlSpec::new(GgmlKernelType::new(GgmlType::Q4_K)?, 1, 300, 8).is_err());
    assert!(GgmlSpec::new(GgmlKernelType::new(GgmlType::Q8_0)?, 1, 48, 8).is_err());
    let fixture = Fixture::new()?;
    let spec = GgmlSpec::new(GgmlKernelType::new(GgmlType::Q6_K)?, 1, 256, 2)?;
    let input = fixture.copy(&vec![bf16::ONE; 256])?;
    let short = fixture.copy(&[0_u8; 210])?;
    let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 2)?;
    let operation = GgmlLinear::compile(&fixture.compiler, spec)?;
    let mut activations = GgmlActivations::new(&fixture.pool, &fixture.stream, 1, 256)?;
    assert!(
        operation
            .execute(&fixture.stream, &input, &mut activations, &short, &mut output)
            .is_err()
    );
    Ok(())
}

struct Fixture {
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    compiler: Compiler,
}

impl Fixture {
    fn new() -> Result<Self> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        Ok(Self {
            stream: context.create_stream()?,
            pool: context.default_memory_pool()?,
            compiler: Compiler::with_include_paths(
                context.clone(),
                [PathBuf::from("/usr/local/cuda/include")],
            )?,
            context,
        })
    }

    fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate::<T>(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }

    fn read<T: DeviceElement + Copy>(&self, values: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        self.stream.copy_to_host(values, &mut host)?;
        Ok(host.to_vec()?)
    }
}
