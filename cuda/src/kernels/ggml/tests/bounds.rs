//! Weights whose blocks end at their last row: one to three single-block
//! rows, so the last block starts at every alignment its size allows (an
//! 18-byte block's third starts 4-byte aligned, its last bytes 2 past a
//! word), in the padded allocation the checkpoint upload makes
//! (`GgmlBlocks`). The decoders' words and the staged GEMV's vectors may run
//! past the last block but not past the padding; `compute-sanitizer --tool
//! memcheck` on these tests checks that, and the values check the decoding.

use mircuda::bf16;

use super::{
    Fixture,
    reference::{self, Blocks, dequantize},
};
use crate::{
    Result,
    kernels::ggml::{GgmlActivations, GgmlDequantize, GgmlKernelType, GgmlLinear, GgmlSpec},
};

#[test]
fn kernels_stop_at_the_weights_last_byte() -> Result<()> {
    let fixture = Fixture::new()?;
    for kind in GgmlKernelType::all() {
        let columns = usize::try_from(kind.ggml_type().block_elements())?;
        for rows in 1..=3 {
            let blocks = Blocks::random(kind, rows, columns, 0x5bd1_e995)?;
            let weight = dequantize(kind, &blocks.bytes, rows, columns)?;
            let device_weight = fixture.copy_blocks(&blocks.bytes)?;
            let mut dense =
                fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, rows * columns)?;
            GgmlDequantize::compile(&fixture.compiler, kind, rows, columns)?
                .execute(&fixture.stream, &device_weight, &mut dense)?;
            for (value, expected) in fixture.read(&dense)?.into_iter().zip(&weight) {
                let value = f64::from(value.to_f32());
                assert!((value - expected).abs() <= expected.abs() / 256.0 + 1e-6, "{kind:?}");
            }
            // One token takes the staged GEMV, sixteen the direct one for
            // types staging fewer.
            for tokens in [1, 16] {
                let input = reference::input(tokens * columns, 0x27d4_eb2f);
                let mut output =
                    fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, tokens * rows)?;
                let mut activations =
                    GgmlActivations::new(&fixture.pool, &fixture.stream, tokens, columns)?;
                GgmlLinear::compile(
                    &fixture.compiler,
                    GgmlSpec::new(kind, tokens, columns, rows)?,
                )?
                .execute(
                    &fixture.stream,
                    &fixture.copy(&input)?,
                    &mut activations,
                    &device_weight,
                    &mut output,
                )?;
                let actual = fixture.read(&output)?;
                for token in 0..tokens {
                    for row in 0..rows {
                        let (expected, magnitude) = reference::dot(
                            &input[token * columns..(token + 1) * columns],
                            &weight[row * columns..(row + 1) * columns],
                        );
                        let value = f64::from(actual[token * rows + row].to_f32());
                        assert!(
                            (value - expected).abs() <= 1e-2f64.mul_add(magnitude, 1e-3),
                            "{kind:?} {rows} rows, {tokens} tokens [{token}, {row}]"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
