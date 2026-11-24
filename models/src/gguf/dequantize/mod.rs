//! Host decoding of GGML blocks: a port of llama.cpp v1.0.0
//! `dequantize_row_*` for every type accelerator kernels decode in place, and
//! the dense float types.
//!
//! It is the reference backend kernels are validated against, not a runtime
//! path: models keep their blocks and never dequantize on the host.

mod iq;
mod k_quants;
mod low_bit;

use super::GgmlType;
use crate::error::{ModelsError, Result};

/// A block decoder appending its values.
type Block = fn(&[u8], &mut Vec<f32>);

/// The values of `rows` consecutive rows of `columns` values each.
pub fn dequantize(ggml_type: GgmlType, bytes: &[u8], columns: usize) -> Result<Vec<f32>> {
    let block: Block = match ggml_type {
        GgmlType::F32 => |bytes, values| {
            values.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        },
        GgmlType::F16 => |bytes, values| values.push(half(bytes)),
        GgmlType::BF16 => |bytes, values| {
            values.push(f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16));
        },
        GgmlType::Q4_0 => |block, values| legacy(block, values, 2, None, false),
        GgmlType::Q4_1 => |block, values| legacy(block, values, 4, None, true),
        GgmlType::Q5_0 => |block, values| legacy(block, values, 6, Some(2), false),
        GgmlType::Q5_1 => |block, values| legacy(block, values, 8, Some(4), true),
        GgmlType::Q8_0 => q8_0,
        GgmlType::Q2_K => k_quants::q2_k,
        GgmlType::Q3_K => k_quants::q3_k,
        GgmlType::Q4_K => k_quants::q4_k,
        GgmlType::Q5_K => k_quants::q5_k,
        GgmlType::Q6_K => k_quants::q6_k,
        GgmlType::IQ4_NL => iq4_nl,
        GgmlType::IQ4_XS => iq4_xs,
        GgmlType::IQ2_XXS => iq::iq2_xxs,
        GgmlType::IQ2_XS => iq::iq2_xs,
        GgmlType::IQ2_S => iq::iq2_s,
        GgmlType::IQ3_XXS => iq::iq3_xxs,
        GgmlType::IQ3_S => iq::iq3_s,
        GgmlType::IQ1_S => iq::iq1_s,
        GgmlType::IQ1_M => iq::iq1_m,
        GgmlType::MXFP4 => mxfp4,
        GgmlType::NVFP4 => low_bit::nvfp4,
        GgmlType::Q1_0 => low_bit::q1_0,
        GgmlType::Q2_0 => low_bit::q2_0,
        GgmlType::TQ1_0 => low_bit::tq1_0,
        GgmlType::TQ2_0 => low_bit::tq2_0,
        _ => {
            return Err(ModelsError::InvalidGguf(format!(
                "no host decoder for GGML type {ggml_type}"
            )));
        },
    };
    let row = usize::try_from(ggml_type.row_bytes(u64::try_from(columns)?)?)?;
    if row == 0 || !bytes.len().is_multiple_of(row) {
        return Err(ModelsError::InvalidGguf(format!(
            "{} bytes are not whole {ggml_type} rows of {columns} values",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / row * columns);
    for chunk in bytes.chunks_exact(usize::try_from(ggml_type.block_bytes())?) {
        block(chunk, &mut values);
    }
    Ok(values)
}

/// IEEE half to single precision.
pub(super) fn half(bytes: &[u8]) -> f32 {
    let bits = u32::from(u16::from_le_bytes([bytes[0], bytes[1]]));
    let sign = (bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0,
        // Subnormal: mantissa · 2^-24, exact in single precision.
        (0, _) => (f32::from(u16::try_from(mantissa).unwrap_or(0)) * 2f32.powi(-24)).to_bits(),
        (0x1F, _) => 0x7F80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// `_q4_0`, `_q4_1`, `_q5_0`, `_q5_1`: qs at `qs`, fifth bits at `qh`, a
/// minimum after d for the `_1` types, else an offset.
fn legacy(block: &[u8], values: &mut Vec<f32>, qs: usize, qh: Option<usize>, minimum: bool) {
    let d = half(block);
    let high = qh.map_or(0, |at| {
        u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
    });
    let (offset, m) = match (minimum, qh) {
        (true, _) => (0, half(&block[2..])),
        (false, Some(_)) => (16, 0.0),
        (false, None) => (8, 0.0),
    };
    let fifth = |j: usize| {
        if qh.is_some() {
            (high >> j) & 1
        } else {
            0
        }
    };
    let value = |nibble: u8, j: usize| {
        let q = i16::from(nibble) | i16::try_from(fifth(j) << 4).unwrap_or(0);
        // Rounded separately, as llama.cpp computes it.
        #[allow(clippy::suboptimal_flops)]
        let value = f32::from(q - offset) * d + m;
        value
    };
    let nibbles = &block[qs..qs + 16];
    values.extend(nibbles.iter().enumerate().map(|(j, byte)| value(byte & 15, j)));
    values.extend(nibbles.iter().enumerate().map(|(j, byte)| value(byte >> 4, j + 16)));
}

fn q8_0(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    values.extend(block[2..34].iter().map(|&q| f32::from(i8::from_le_bytes([q])) * d));
}

const IQ4_NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn table(qs: &[u8], table: &[i8; 16], d: f32, values: &mut Vec<f32>) {
    values.extend(qs.iter().map(|q| f32::from(table[usize::from(q & 15)]) * d));
    values.extend(qs.iter().map(|q| f32::from(table[usize::from(q >> 4)]) * d));
}

fn iq4_nl(block: &[u8], values: &mut Vec<f32>) {
    table(&block[2..18], &IQ4_NL, half(block), values);
}

fn iq4_xs(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    for ib in 0..8 {
        let low = (block[4 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        let high = u8::try_from((scales_h >> (2 * ib)) & 3).unwrap_or(0);
        let scale = f32::from(i16::from(low | (high << 4)) - 32);
        table(&block[8 + 16 * ib..24 + 16 * ib], &IQ4_NL, d * scale, values);
    }
}

/// `GGML_E8M0_TO_FP32_HALF`: 2^(e - 128), with the table doubled.
fn mxfp4(block: &[u8], values: &mut Vec<f32>) {
    let d = 2f32.powi(i32::from(block[0]) - 128);
    table(&block[1..17], &MXFP4, d, values);
}

#[cfg(test)]
mod tests;
