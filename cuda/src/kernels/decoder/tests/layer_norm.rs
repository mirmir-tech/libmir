//! The mean-centred norm variants: `LayerNorm` without bias (Cohere) and
//! with its bias packed after the weight (Phi-3.5-MoE).

use mircuda::bf16;

use super::{NormFormula, Result, RmsNorm, runtime::Runtime};

const COLUMNS: usize = 96;

#[test]
fn mean_centred_norm_matches_cpu_layer_norm() -> Result<()> {
    check(NormFormula::MeanCentred)
}

#[test]
fn biased_mean_centred_norm_adds_the_packed_bias() -> Result<()> {
    check(NormFormula::MeanCentredBiased)
}

fn check(formula: NormFormula) -> Result<()> {
    let runtime = Runtime::new()?;
    // Rows offset far from zero, which RMS would not remove.
    let values = (0..2 * COLUMNS)
        .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index % 11)?) + 20.0)))
        .collect::<Result<Vec<_>>>()?;
    let weights = (0..COLUMNS)
        .map(|index| Ok(f32::from(u16::try_from(index % 5)?) * 0.25 + 0.5))
        .collect::<Result<Vec<_>>>()?;
    let biases = (0..COLUMNS)
        .map(|index| Ok(f32::from(u16::try_from(index % 7)?) * 0.5 - 1.5))
        .collect::<Result<Vec<_>>>()?;
    let with_bias = formula == NormFormula::MeanCentredBiased;
    let packed: Vec<bf16> = weights
        .iter()
        .chain(biases.iter().filter(|_| with_bias))
        .map(|&value| bf16::from_f32(value))
        .collect();
    let input = runtime.copy(&values)?;
    let weight = runtime.copy(&packed)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, values.len())?;
    RmsNorm::compile_formula(&runtime.compiler, 2, COLUMNS, 1.0e-5, formula)?
        .execute(&runtime.stream, &input, &weight, &mut output)?;
    let actual = runtime.read(&output)?;
    for (row, chunk) in values.chunks(COLUMNS).enumerate() {
        let floats: Vec<f32> = chunk.iter().map(|value| value.to_f32()).collect();
        let mean = floats.iter().sum::<f32>() / 96.0;
        let variance = floats.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / 96.0;
        let inverse = (variance + 1.0e-5).sqrt().recip();
        for (column, value) in floats.iter().enumerate() {
            let bias = if with_bias {
                biases[column]
            } else {
                0.0
            };
            let expected = ((value - mean) * inverse).mul_add(weights[column], bias);
            let actual = actual[row * COLUMNS + column].to_f32();
            assert!(
                (actual - expected).abs() <= 1.0e-2 * expected.abs().max(1.0),
                "{actual} {expected}"
            );
        }
    }
    Ok(())
}
