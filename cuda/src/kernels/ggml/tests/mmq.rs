//! Grouped GGML GEMM against the reference dequantization.

use mircuda::bf16;

use super::{
    Fixture, KINDS,
    reference::{self, Blocks, dequantize},
};
use crate::{
    Result,
    kernels::ggml::{GgmlGroupedMmq, GgmlMmqGroups, GgmlMmqInput, GgmlMmqPlacement},
};

const EXPERTS: usize = 3;
/// Weight rows per expert: one full 64-column tile and a partial one.
const ROWS: usize = 72;

#[test]
fn grouped_mmq_matches_the_reference_per_expert() -> Result<()> {
    let fixture = Fixture::new()?;
    // Expert 0 takes no rows, expert 1 a full tile and a ragged one,
    // expert 2 five rows.
    let counts = [0_usize, 70, 5];
    let offsets = [0_u32, 0, 70];
    let total = counts.iter().sum::<usize>();
    // Outputs interleave at odd columns of rows twice as wide.
    let placement = GgmlMmqPlacement { ld: 2 * ROWS, stride: 2, offset: 1 };
    for (kind, columns) in KINDS.into_iter().flat_map(|kind| [(kind, 512), (kind, 768)]) {
        let blocks = Blocks::random(kind, EXPERTS * ROWS, columns, 0x2545_f491)?;
        let weight = dequantize(kind, &blocks.bytes, EXPERTS * ROWS, columns);
        let input = reference::input(total * columns, 0x9e37_79b9);
        let mut operation = GgmlGroupedMmq::compile(
            &fixture.compiler,
            &fixture.pool,
            &fixture.stream,
            &[kind],
            EXPERTS,
            ROWS,
            columns,
            total,
        )?;
        let mut activations = operation.activations(&fixture.pool, &fixture.stream)?;
        let device_offsets = fixture.copy(&offsets)?;
        let device_counts =
            fixture.copy(&counts.map(u32::try_from).map(|count| count.unwrap_or(0)))?;
        let groups = GgmlMmqGroups {
            offsets: &device_offsets,
            counts: &device_counts,
        };
        let device_input = fixture.copy(&input)?;
        let device_weight = fixture.copy(&blocks.bytes)?;
        let mut output = fixture.copy(&vec![bf16::ZERO; total * placement.ld])?;
        operation.prepare(
            &fixture.stream,
            GgmlMmqInput::Rows(&device_input),
            total,
            groups,
            &mut activations,
        )?;
        operation.multiply(
            &fixture.stream, &activations, groups, kind, &device_weight, &mut output, placement,
        )?;
        let actual = fixture.read(&output)?;
        for expert in 0..EXPERTS {
            let first = usize::try_from(offsets[expert])?;
            for row in first..first + counts[expert] {
                for column in 0..ROWS {
                    let at = (expert * ROWS + column) * columns;
                    let (expected, magnitude) = reference::dot(
                        &input[row * columns..(row + 1) * columns],
                        &weight[at..at + columns],
                    );
                    let value = f64::from(actual[row * placement.ld + 2 * column + 1].to_f32());
                    assert!(
                        (value - expected).abs() <= 1e-2f64.mul_add(magnitude, 1e-3),
                        "{kind:?} columns {columns} expert {expert} [{row}, {column}]: \
                         {value} vs {expected}"
                    );
                }
            }
        }
        // Even columns stay untouched.
        assert!(actual.iter().step_by(2).all(|value| value.to_f32() == 0.0));
    }
    Ok(())
}
