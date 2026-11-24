//! Routed GGML products against the host decoder.

use models::gguf::{GgmlType, dequantize};

use super::GgmlLinear;
use crate::engine::{Array, Dtype, Result, Stream, kernels::ggml::GgmlMatrix};

const EXPERTS: usize = 3;
const ROWS: usize = 16;
const COLUMNS: usize = 64;

/// `Q8_0` blocks with small finite scales.
fn bank() -> Vec<u8> {
    let mut state = 0x9E37_79B9_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut bytes = Vec::new();
    for _ in 0..EXPERTS * ROWS * COLUMNS / 32 {
        bytes.extend_from_slice(
            &(0x2000_u16 + u16::try_from(next() % 512).unwrap_or(0)).to_le_bytes(),
        );
        bytes.extend((0..32).map(|_| next().to_le_bytes()[1]));
    }
    bytes
}

fn linear(bytes: &[u8], stream: &Stream) -> Result<GgmlLinear> {
    let values = bytes.iter().map(|&byte| u32::from(byte)).collect::<Vec<_>>();
    let rows = i32::try_from(EXPERTS * ROWS)?;
    let weight = Array::from_u32(&values, &[rows, i32::try_from(bytes.len())? / rows])?
        .astype(Dtype::Uint8, stream)?;
    let matrix = GgmlMatrix::new(GgmlType::Q8_0, COLUMNS, &weight)?;
    Ok(GgmlLinear {
        weight,
        matrix,
        layout: super::GgmlLayout::Bank { experts: EXPERTS, scale: None },
    })
}

#[test]
fn gathered_and_dense_paths_match_the_host_decoder() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let bytes = bank();
    let host = dequantize(GgmlType::Q8_0, &bytes, COLUMNS)?;
    let linear = linear(&bytes, &stream)?;
    // Few assignments take the gathered GEMV, many the dense `gather_mm`.
    for (tokens, top_k) in [(3_usize, 2_usize), (300, 2)] {
        let input = (0..tokens * COLUMNS)
            .map(|index| f32::from(u16::try_from(index % 23).unwrap_or(0)) / 23.0 - 0.5)
            .collect::<Vec<_>>();
        let selected = (0..tokens * top_k)
            .map(|index| u32::try_from((index * 7 + index / 5) % EXPERTS).unwrap_or(0))
            .collect::<Vec<_>>();
        let x = Array::from_f32(&input, &[i32::try_from(tokens)?, 1, 1, i32::try_from(COLUMNS)?])?
            .astype(Dtype::Bfloat16, &stream)?;
        let rounded = x.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?;
        let indices = Array::from_u32(&selected, &[i32::try_from(tokens)?, i32::try_from(top_k)?])?;
        let output = linear.gather(&x, &indices, false, &stream)?;
        assert_eq!(output.shape()?, [i32::try_from(tokens)?, i32::try_from(top_k)?, 1, 16]);
        let output = output.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?;
        for (assignment, &expert) in selected.iter().enumerate() {
            let x = &rounded[assignment / top_k * COLUMNS..][..COLUMNS];
            for row in 0..ROWS {
                let w = &host[(usize::try_from(expert)? * ROWS + row) * COLUMNS..][..COLUMNS];
                let (dot, magnitude) =
                    x.iter().zip(w).fold((0.0_f32, 0.0_f32), |(d, m), (x, w)| {
                        (x.mul_add(*w, d), (x * w).abs() + m)
                    });
                let actual = output[assignment * ROWS + row];
                assert!(
                    (actual - dot).abs() <= magnitude.mul_add(1.0e-2, 1.0e-3),
                    "{tokens} tokens, assignment {assignment}, row {row}: {actual} != {dot}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn grouped_sorted_assignments_match_the_host_decoder() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let bytes = bank();
    let host = dequantize(GgmlType::Q8_0, &bytes, COLUMNS)?;
    let linear = linear(&bytes, &stream)?;
    // Sorted with uneven runs, including an absent expert and a 16-row tile
    // that spans two experts.
    let counts = [517_usize, 0, 290];
    let selected = counts
        .iter()
        .enumerate()
        .flat_map(|(expert, &count)| std::iter::repeat_n(u32::try_from(expert).unwrap_or(0), count))
        .collect::<Vec<_>>();
    let assignments = selected.len();
    let input = (0..assignments * COLUMNS)
        .map(|index| f32::from(u16::try_from(index % 29).unwrap_or(0)) / 29.0 - 0.5)
        .collect::<Vec<_>>();
    let x = Array::from_f32(&input, &[i32::try_from(assignments)?, 1, i32::try_from(COLUMNS)?])?
        .astype(Dtype::Bfloat16, &stream)?;
    let rounded = x.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?;
    let indices = Array::from_u32(&selected, &[i32::try_from(assignments)?])?;
    // This many assignments per expert take the dequantized bank; the grouped
    // kernel serves sparser batches, so it is launched directly as well.
    let banked = linear.gather(&x, &indices, true, &stream)?;
    assert_eq!(banked.shape()?, [i32::try_from(assignments)?, 1, 16]);
    let flat = x.reshape(&[i32::try_from(assignments)?, i32::try_from(COLUMNS)?], &stream)?;
    let grouped = stream.kernels().ggml_grouped(
        &stream,
        [&flat, &linear.weight, &indices],
        linear.matrix,
        EXPERTS,
    )?;
    for output in [banked, grouped] {
        let output = output.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?;
        assert_matches_host(&selected, &rounded, &host, &output)?;
    }
    Ok(())
}

fn assert_matches_host(
    selected: &[u32],
    rounded: &[f32],
    host: &[f32],
    output: &[f32],
) -> Result<()> {
    for (assignment, &expert) in selected.iter().enumerate() {
        let x = &rounded[assignment * COLUMNS..][..COLUMNS];
        for row in 0..ROWS {
            let w = &host[(usize::try_from(expert)? * ROWS + row) * COLUMNS..][..COLUMNS];
            let (dot, magnitude) = x
                .iter()
                .zip(w)
                .fold((0.0_f32, 0.0_f32), |(d, m), (x, w)| (x.mul_add(*w, d), (x * w).abs() + m));
            let actual = output[assignment * ROWS + row];
            assert!(
                (actual - dot).abs() <= magnitude.mul_add(1.0e-2, 1.0e-3),
                "assignment {assignment}, row {row}: {actual} != {dot}"
            );
        }
    }
    Ok(())
}

#[test]
fn output_scales_multiply_each_experts_products() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let bytes = bank();
    let plain = linear(&bytes, &stream)?;
    let mut scaled = linear(&bytes, &stream)?;
    scaled.layout = super::GgmlLayout::Bank {
        experts: EXPERTS,
        scale: Some(Array::from_f32(&[0.5, 2.0, -3.0], &[3])?),
    };
    let input = (0..2 * COLUMNS)
        .map(|index| f32::from(u16::try_from(index % 19).unwrap_or(0)) / 19.0 - 0.5)
        .collect::<Vec<_>>();
    let x = Array::from_f32(&input, &[2, 1, 1, i32::try_from(COLUMNS)?])?
        .astype(Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[2, 0, 1, 2], &[2, 2])?;
    let values = |linear: &GgmlLinear| -> Result<Vec<f32>> {
        linear
            .gather(&x, &indices, false, &stream)?
            .astype(Dtype::Float32, &stream)?
            .to_vec_f32(&stream)
    };
    let (plain, scaled) = (values(&plain)?, values(&scaled)?);
    for (assignment, factor) in [-3.0_f32, 0.5, 2.0, -3.0].into_iter().enumerate() {
        for row in 0..ROWS {
            let (expected, actual) =
                (plain[assignment * ROWS + row] * factor, scaled[assignment * ROWS + row]);
            assert!(
                (expected - actual).abs() <= expected.abs().mul_add(1.0e-2, 1.0e-3),
                "{expected} != {actual}"
            );
        }
    }
    Ok(())
}
