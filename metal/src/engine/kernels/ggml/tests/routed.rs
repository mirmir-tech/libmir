//! Routed GGML products of every type against the host decoder, including
//! expert widths that are not whole K-quant super-blocks (GLM-4.5-Air's 1408)
//! and widths whose last 128-column step runs past the row (704), which the
//! grouped kernel must not read beyond.

use super::{GGML_METAL_TYPES, GgmlMatrix, Random, blocks, close, host, weight};
use crate::engine::{Array, Dtype, Result, Stream, kernels::Kernels};

const EXPERTS: usize = 3;
const ROWS: usize = 8;

/// `output[a]` against `input[row_of(a)] · expert[indices[a]]ᵀ`.
fn check(
    label: &str,
    output: &[f32],
    rounded: &[f32],
    expected: &[f32],
    indices: &[u32],
    row_of: impl Fn(usize) -> usize,
    columns: usize,
) -> Result<()> {
    for (assignment, &expert) in indices.iter().enumerate() {
        let x = &rounded[row_of(assignment) * columns..][..columns];
        for row in 0..ROWS {
            let w = &expected[(usize::try_from(expert)? * ROWS + row) * columns..][..columns];
            let (dot, magnitude) = x.iter().zip(w).fold((0.0, 0.0), |(dot, sum), (x, w)| {
                let term = f64::from(*x) * f64::from(*w);
                (dot + term, sum + term.abs())
            });
            let actual = output[assignment * ROWS + row];
            assert!(
                close(actual, dot, magnitude),
                "{label} a{assignment} r{row}: {actual} != {dot}"
            );
        }
    }
    Ok(())
}

fn input(rows: usize, columns: usize, seed: u64, stream: &Stream) -> Result<(Array, Vec<f32>)> {
    let mut random = Random(seed);
    let values = (0..rows * columns)
        .map(|_| random.unit().mul_add(2.0, -1.0))
        .collect::<Vec<_>>();
    let input = Array::from_f32(&values, &[i32::try_from(rows)?, i32::try_from(columns)?])?
        .astype(Dtype::Bfloat16, stream)?;
    let rounded = input.astype(Dtype::Float32, stream)?.to_vec_f32(stream)?;
    Ok((input, rounded))
}

#[test]
fn gathered_and_grouped_products_match_the_host_decoder() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for columns in [512_usize, 704, 1408] {
        for (index, &ggml_type) in GGML_METAL_TYPES.iter().enumerate() {
            if u64::try_from(columns)? % ggml_type.block_elements() != 0 {
                continue;
            }
            let data = blocks(ggml_type, EXPERTS * ROWS, columns, 211 + u64::try_from(index)?)?;
            let expected = host(ggml_type, &data, columns)?;
            let bank = weight(&data, EXPERTS * ROWS, &stream)?;
            let matrix = GgmlMatrix::new(ggml_type, columns, &bank)?;
            let label = format!("{ggml_type} {columns}");

            // Four tokens, two experts each: assignment `a` reads row `a / 2`.
            let (tokens, rounded) = input(4, columns, 5, &stream)?;
            let selected = [2_u32, 0, 1, 2, 0, 0, 1, 2];
            let indices = Array::from_u32(&selected, &[8])?;
            let gathered = stream.kernels().ggml_gathered(
                &stream,
                [&tokens, &bank, &indices],
                matrix,
                EXPERTS,
            )?;
            check(
                &format!("gathered {label}"),
                &gathered.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?,
                &rounded,
                &expected,
                &selected,
                |assignment| assignment / 2,
                columns,
            )?;

            // Expert-sorted assignments with an absent expert, one row each.
            let sorted = [[0_u32; 21].as_slice(), &[2; 19]].concat();
            let (rows, rounded) = input(sorted.len(), columns, 9, &stream)?;
            let indices = Array::from_u32(&sorted, &[i32::try_from(sorted.len())?])?;
            let grouped = stream.kernels().ggml_grouped(
                &stream,
                [&rows, &bank, &indices],
                matrix,
                EXPERTS,
            )?;
            check(
                &format!("grouped {label}"),
                &grouped.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?,
                &rounded,
                &expected,
                &sorted,
                |assignment| assignment,
                columns,
            )?;
        }
    }
    Ok(())
}

/// A bank of no experts or one that does not divide into them, and indices
/// of another type than Uint32, are refused before any kernel runs.
#[test]
fn routed_products_refuse_bad_banks_and_indices() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let ggml_type = GGML_METAL_TYPES[0];
    let data = blocks(ggml_type, EXPERTS * ROWS, 512, 3)?;
    let bank = weight(&data, EXPERTS * ROWS, &stream)?;
    let matrix = GgmlMatrix::new(ggml_type, 512, &bank)?;
    let (tokens, _) = input(2, 512, 1, &stream)?;
    let unsigned = Array::from_u32(&[0, 1], &[2])?;
    let signed = unsigned.astype(Dtype::Int32, &stream)?;
    let kernels = stream.kernels();
    for experts in [0, 5] {
        assert!(
            kernels
                .ggml_gathered(&stream, [&tokens, &bank, &unsigned], matrix, experts)
                .is_err()
        );
        assert!(
            kernels
                .ggml_grouped(&stream, [&tokens, &bank, &unsigned], matrix, experts)
                .is_err()
        );
    }
    assert!(
        kernels
            .ggml_gathered(&stream, [&tokens, &bank, &signed], matrix, EXPERTS)
            .is_err()
    );
    assert!(
        kernels
            .ggml_grouped(&stream, [&tokens, &bank, &signed], matrix, EXPERTS)
            .is_err()
    );
    // No assignment is no product: a zero group over an empty grid.
    let none = Array::from_u32(&[], &[0])?;
    for launch in [Kernels::ggml_gathered, Kernels::ggml_grouped] {
        assert!(launch(kernels, &stream, [&tokens, &bank, &none], matrix, EXPERTS).is_err());
    }
    assert!(
        kernels
            .ggml_grouped(&stream, [&tokens, &bank, &unsigned], matrix, EXPERTS)
            .is_ok()
    );
    Ok(())
}
