//! CUDA logits across row lengths around the 64-token attention blocks and
//! the local/full attention transitions, against the CPU backend.

use libmir::decision::{DecisionBackend, DecisionModel, DecisionPrecision, DecisionRow};

use super::{BF16_TOLERANCE, F32_TOLERANCE, Failure, reference, rows};

const LENGTHS: [usize; 19] =
    [1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 512, 513, 1023, 1024];

/// The last `length` tokens of `row`, keeping the markers still inside it
/// and the final token when none is.
fn cut(row: &DecisionRow, length: usize) -> DecisionRow {
    let dropped = row.tokens.len().saturating_sub(length);
    let mut markers: Vec<usize> = row
        .markers
        .iter()
        .filter(|&&marker| marker >= dropped)
        .map(|marker| marker - dropped)
        .collect();
    if markers.is_empty() {
        markers.push(length - 1);
    }
    DecisionRow {
        tokens: row.tokens[dropped..].to_vec(),
        markers,
        ..row.clone()
    }
}

fn largest(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .flat_map(|(actual, expected)| {
            assert_eq!(actual.len(), expected.len(), "option count");
            actual.iter().zip(expected).map(|(actual, expected)| (actual - expected).abs())
        })
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs a GPU and the Laya checkpoint in LIBMIR_LAYA_MODEL; run `make laya-parity-cuda`"]
#[expect(clippy::print_stdout, reason = "the measured differences are the evidence")]
fn cuda_matches_cpu_across_row_lengths() -> Result<(), Failure> {
    let root = std::env::var_os("LIBMIR_LAYA_MODEL")
        .ok_or("LIBMIR_LAYA_MODEL must name the Laya checkpoint directory")?;
    let cpu = DecisionModel::load(&root, DecisionBackend::Cpu)?;
    let reference = reference()?;
    let long = rows(&cpu, &reference)?
        .into_iter()
        .flat_map(|(_, rows)| rows)
        .max_by_key(|row| row.tokens.len())
        .ok_or("the fixture has no rows")?;
    assert_eq!(long.tokens.len(), 1024, "the sweep needs a full-length row");
    let sweep: Vec<DecisionRow> = LENGTHS.iter().map(|&length| cut(&long, length)).collect();
    let expected = cpu.logits(&sweep)?;
    drop(cpu);
    for (precision, tolerance) in
        [(DecisionPrecision::F32, F32_TOLERANCE), (DecisionPrecision::Bf16, BF16_TOLERANCE)]
    {
        let model = DecisionModel::load(&root, DecisionBackend::Cuda(precision))?;
        let batched = largest(&model.logits(&sweep)?, &expected);
        let mut single = 0.0_f32;
        for (row, expected) in sweep.iter().zip(&expected) {
            single = single.max(largest(
                &model.logits(std::slice::from_ref(row))?,
                std::slice::from_ref(expected),
            ));
        }
        println!(
            "{precision:?}: lengths {LENGTHS:?}: max |Δlogit| vs CPU {batched:.3e} in one ragged batch, {single:.3e} row by row"
        );
        assert!(batched.max(single) <= tolerance, "{precision:?}: {batched} / {single}");
    }
    Ok(())
}
