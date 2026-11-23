//! Routed-expert GEMV and combine against the reference dequantization.

use mircuda::bf16;
use models::gguf::GgmlType;

use super::{
    Fixture,
    reference::{self, Blocks, dequantize},
};
use crate::{
    Result,
    kernels::ggml::{
        GgmlActivations, GgmlBankRows, GgmlExpertSpec, GgmlGatheredLinear, GgmlKernelType,
        GgmlSelections,
    },
};

const EXPERTS: usize = 5;
const ROWS: usize = 24;
const COLUMNS: usize = 512;

#[test]
fn gathered_kernels_match_the_reference_per_selection() -> Result<()> {
    let fixture = Fixture::new()?;
    // Three tokens, two selections each; expert 3 is chosen twice.
    let (tokens, top_k) = (3, 2);
    let selected = [3_u32, 0, 4, 3, 1, 2];
    for kind in GgmlKernelType::all() {
        let blocks = Blocks::random(kind, EXPERTS * ROWS, COLUMNS, 0x1656_67b1)?;
        let weight = dequantize(kind, &blocks.bytes, EXPERTS * ROWS, COLUMNS);
        // The whole matrix, and the second half of every expert's rows as
        // the up projection of a fused gate/up bank reads it.
        for (rows, bank) in [
            (ROWS, GgmlBankRows::whole(ROWS)),
            (ROWS / 2, GgmlBankRows { expert_rows: ROWS, first_row: ROWS / 2 }),
        ] {
            let spec = GgmlExpertSpec {
                kind,
                experts: EXPERTS,
                rows,
                columns: COLUMNS,
            };
            let operation = GgmlGatheredLinear::compile(&fixture.compiler, spec)?;
            let input = reference::input(tokens * COLUMNS, 0xd3a2_646c);
            let device_input = fixture.copy(&input)?;
            let device_weight = fixture.copy(&blocks.bytes)?;
            let device_selected = fixture.copy(&selected)?;
            let mut activations =
                GgmlActivations::new(&fixture.pool, &fixture.stream, tokens, COLUMNS)?;
            let mut output =
                fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, selected.len() * rows)?;
            operation.quantize(&fixture.stream, &device_input, &mut activations, tokens)?;
            let selections = GgmlSelections {
                selected: &device_selected,
                count: selected.len(),
                group: top_k,
            };
            operation.multiply(
                &fixture.stream,
                &activations,
                (&device_weight, bank),
                &selections,
                &mut output,
            )?;
            let actual = fixture.read(&output)?;
            for (selection, &expert) in selected.iter().enumerate() {
                let token = selection / top_k;
                for row in 0..rows {
                    let at = (usize::try_from(expert)? * ROWS + bank.first_row + row) * COLUMNS;
                    let (expected, magnitude) = reference::dot(
                        &input[token * COLUMNS..(token + 1) * COLUMNS],
                        &weight[at..at + COLUMNS],
                    );
                    let value = f64::from(actual[selection * rows + row].to_f32());
                    assert!(
                        (value - expected).abs() <= 1e-2f64.mul_add(magnitude, 1e-3),
                        "{kind:?} selection {selection} row {row}: {value} vs {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn combine_sums_selections_by_routing_weight() -> Result<()> {
    let fixture = Fixture::new()?;
    let (tokens, top_k) = (2, 3);
    let spec = GgmlExpertSpec {
        kind: GgmlKernelType::new(GgmlType::Q8_0)?,
        experts: EXPERTS,
        rows: ROWS,
        columns: COLUMNS,
    };
    let operation = GgmlGatheredLinear::compile(&fixture.compiler, spec)?;
    let partial = reference::input(tokens * top_k * ROWS, 0x9e37_79b9);
    let routing = [0.5_f32, 0.25, 0.25, 0.125, 0.375, 0.5].map(bf16::from_f32);
    let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, tokens * ROWS)?;
    operation.combine(
        &fixture.stream,
        &fixture.copy(&partial)?,
        &fixture.copy(&routing)?,
        &mut output,
        top_k,
    )?;
    let actual = fixture.read(&output)?;
    for token in 0..tokens {
        for row in 0..ROWS {
            let expected = (0..top_k)
                .map(|k| {
                    let selection = token * top_k + k;
                    routing[selection].to_f32() * partial[selection * ROWS + row].to_f32()
                })
                .sum::<f32>();
            let value = actual[token * ROWS + row].to_f32();
            assert!(
                (value - expected).abs() <= expected.abs() / 128.0 + 1e-6,
                "{value} vs {expected}"
            );
        }
    }
    Ok(())
}
