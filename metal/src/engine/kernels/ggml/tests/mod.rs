//! Metal GGML kernels against the host port of llama.cpp `dequantize_row_*`.

mod bandwidth;
mod chain;
mod routed;

use models::gguf::{GgmlType, dequantize};

use super::{GGML_GEMV_MAX_TOKENS, GGML_METAL_TYPES, GgmlMatrix};
use crate::engine::{Array, Dtype, Result, Stream};

pub(super) struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        f32::from(u16::try_from(self.next() >> 48).unwrap_or(0)) / 65_535.0
    }
}

/// Random blocks whose half scales are small and finite.
pub(super) fn blocks(
    ggml_type: GgmlType,
    rows: usize,
    columns: usize,
    seed: u64,
) -> Result<Vec<u8>> {
    let mut random = Random(seed | 1);
    let bytes = usize::try_from(ggml_type.row_bytes(u64::try_from(columns)?)?)? * rows;
    let mut data = (0..bytes).map(|_| random.next().to_le_bytes()[3]).collect::<Vec<_>>();
    let scales: &[usize] = match ggml_type {
        GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        GgmlType::MXFP4 | GgmlType::NVFP4 | GgmlType::IQ1_M => &[],
        GgmlType::TQ1_0 => &[52],
        GgmlType::TQ2_0 => &[64],
        _ => &[0],
    };
    for block in data.chunks_exact_mut(usize::try_from(ggml_type.block_bytes())?) {
        for &at in scales {
            let scale = half_bits(random.unit().mul_add(0.01, 0.001));
            block[at..at + 2].copy_from_slice(&scale.to_le_bytes());
        }
        if ggml_type == GgmlType::MXFP4 {
            block[0] = 118 + random.next().to_le_bytes()[0] % 8;
        }
        // Small UE4M3 group scales (exponents 1..4).
        if ggml_type == GgmlType::NVFP4 {
            for scale in &mut block[..4] {
                *scale = 8 + random.next().to_le_bytes()[0] % 32;
            }
        }
        // IQ1_M has no d field: its half is the top nibbles of scales[0..4].
        if ggml_type == GgmlType::IQ1_M {
            let d = half_bits(random.unit().mul_add(0.01, 0.001));
            for (k, at) in (48..56).step_by(2).enumerate() {
                let nibble = (d >> (4 * k)) & 0xF;
                let value =
                    (u16::from_le_bytes([block[at], block[at + 1]]) & 0x0FFF) | (nibble << 12);
                block[at..at + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(data)
}

/// A positive normal value in half precision, truncated.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = (bits >> 23 & 0xFF) - 112;
    u16::try_from(exponent << 10 | (bits >> 13 & 0x3FF)).unwrap_or(0)
}

pub(super) fn weight(data: &[u8], rows: usize, stream: &Stream) -> Result<Array> {
    let values = data.iter().map(|&byte| u32::from(byte)).collect::<Vec<_>>();
    let row_bytes = i32::try_from(data.len() / rows)?;
    Array::from_u32(&values, &[i32::try_from(rows)?, row_bytes])?.astype(Dtype::Uint8, stream)
}

/// The host values of whole rows through the shared `models` decoder.
pub(super) fn host(ggml_type: GgmlType, data: &[u8], columns: usize) -> Result<Vec<f32>> {
    Ok(dequantize(ggml_type, data, columns)?)
}

pub(super) fn close(actual: f32, expected: f64, magnitude: f64) -> bool {
    (f64::from(actual) - expected).abs() <= magnitude.mul_add(1.0e-2, 1.0e-6)
}

#[test]
fn dequantization_and_embedding_match_the_host_decoder() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let (rows, columns) = (5, 512);
    for (index, &ggml_type) in GGML_METAL_TYPES.iter().enumerate() {
        let data = blocks(ggml_type, rows, columns, 7 + u64::try_from(index)?)?;
        let expected = host(ggml_type, &data, columns)?;
        let weight = weight(&data, rows, &stream)?;
        let matrix = GgmlMatrix::new(ggml_type, columns, &weight)?;
        let dense =
            stream
                .kernels()
                .ggml_dequantize(&stream, &weight, matrix, mirtal::DType::Float32)?;
        let dense = dense.to_vec_f32(&stream)?;
        for (actual, expected) in dense.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= expected.abs() * 1.0e-6,
                "{ggml_type}: {actual} != {expected}"
            );
        }

        let ids = Array::from_u32(&[3, 0, 9, 3], &[2, 2])?;
        let rows_out = stream.kernels().ggml_embedding(
            &stream,
            &weight,
            &ids,
            matrix,
            mirtal::DType::Float32,
        )?;
        assert_eq!(rows_out.shape()?, [2, 2, 512]);
        let rows_out = rows_out.to_vec_f32(&stream)?;
        for (slot, id) in [3_usize, 0, 9, 3].into_iter().enumerate() {
            let actual = &rows_out[slot * columns..(slot + 1) * columns];
            if id >= rows {
                assert!(actual.iter().all(|value| *value == 0.0));
            } else {
                assert_eq!(actual, &dense[id * columns..(id + 1) * columns], "{ggml_type}");
            }
        }
    }
    Ok(())
}

#[test]
fn gemv_matches_the_host_dot_product() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for (columns, rows) in [(512, 13), (2048, 3), (1408, 7), (10944, 3)] {
        for (index, &ggml_type) in GGML_METAL_TYPES.iter().enumerate() {
            if u64::try_from(columns)? % ggml_type.block_elements() != 0 {
                continue;
            }
            let data = blocks(ggml_type, rows, columns, 31 + u64::try_from(index)?)?;
            let expected = host(ggml_type, &data, columns)?;
            let weight = weight(&data, rows, &stream)?;
            let matrix = GgmlMatrix::new(ggml_type, columns, &weight)?;
            for tokens in [1, 2, 5, 8, 13, 17, GGML_GEMV_MAX_TOKENS] {
                let mut random = Random(97 + u64::try_from(tokens)?);
                let input = (0..tokens * columns)
                    .map(|_| random.unit().mul_add(2.0, -1.0))
                    .collect::<Vec<_>>();
                let input =
                    Array::from_f32(&input, &[i32::try_from(tokens)?, i32::try_from(columns)?])?
                        .astype(Dtype::Bfloat16, &stream)?;
                let rounded = input.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?;
                let output = stream.kernels().ggml_gemv(&stream, &input, &weight, matrix)?;
                assert_eq!(output.shape()?, [i32::try_from(tokens)?, i32::try_from(rows)?]);
                let output = output.astype(Dtype::Float32, &stream)?.to_vec_f32(&stream)?;
                for token in 0..tokens {
                    for row in 0..rows {
                        let x = &rounded[token * columns..(token + 1) * columns];
                        let w = &expected[row * columns..(row + 1) * columns];
                        let (dot, magnitude) =
                            x.iter().zip(w).fold((0.0, 0.0), |(dot, sum), (x, w)| {
                                let term = f64::from(*x) * f64::from(*w);
                                (dot + term, sum + term.abs())
                            });
                        let actual = output[token * rows + row];
                        assert!(
                            close(actual, dot, magnitude),
                            "{ggml_type} {columns}x{rows} t{tokens}: {actual} != {dot}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_types_without_a_decoder_and_partial_rows() -> Result<()> {
    let stream = Stream::new_gpu()?;
    // Q8_K is llama.cpp's activation type: no file stores weights in it.
    let weight = weight(&[0; 292], 1, &stream)?;
    assert!(GgmlMatrix::new(GgmlType::Q8_K, 256, &weight).is_err());
    assert!(GgmlMatrix::new(GgmlType::Q4_K, 256, &weight).is_err());
    Ok(())
}

#[test]
fn mxfp4_repack_keeps_every_value() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let (rows, columns) = (3, 96);
    let data = blocks(GgmlType::MXFP4, rows, columns, 11)?;
    let expected = dequantize(GgmlType::MXFP4, &data, columns)?;
    let [nibbles, scales] =
        stream.kernels().ggml_mxfp4_repack(&stream, &weight(&data, rows, &stream)?)?;
    assert_eq!(nibbles.shape()?, [3, 3, 16]);
    assert_eq!(scales.shape()?, [3, 3]);
    let (nibbles, scales) = (nibbles.to_vec_u32(&stream)?, scales.to_vec_u32(&stream)?);
    // Code 8 is negative zero; GGML's table decodes it as zero.
    let e2m1: [f32; 16] =
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    for (index, expected) in expected.iter().enumerate() {
        let block = index / 32;
        let byte = nibbles[block * 16 + index % 32 / 2];
        let code = if index % 2 == 0 {
            byte & 15
        } else {
            byte >> 4
        };
        let scale = 2_f32.powi(i32::try_from(scales[block])? - 127);
        let actual = e2m1[usize::try_from(code)?] * scale;
        assert_eq!(actual.to_bits(), expected.to_bits(), "value {index}");
    }
    Ok(())
}
