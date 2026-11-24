//! llama.cpp's one- and two-bit blocks (`dequantize_row_q1_0`, `_q2_0`) and
//! the ternary `TQ1_0` and `TQ2_0` (`dequantize_row_tq1_0`, `_tq2_0`), and
//! NVFP4 (`dequantize_row_nvfp4`).

use super::{MXFP4, half};

/// `Q1_0`: half d, 128 sign bits; set bits are d, clear bits -d.
pub(super) fn q1_0(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    values.extend((0..128).map(|j| {
        if block[2 + j / 8] >> (j % 8) & 1 == 1 {
            d
        } else {
            -d
        }
    }));
}

/// `Q2_0`: half d, 64 two-bit values in `{-1, 0, 1, 2}`.
pub(super) fn q2_0(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    values.extend((0..64).map(|j| (f32::from(block[2 + j / 4] >> (2 * (j % 4)) & 3) - 1.0) * d));
}

/// `TQ2_0`: qs[64] of two-bit ternary digits, half d.
pub(super) fn tq2_0(block: &[u8], values: &mut Vec<f32>) {
    let d = half(&block[64..]);
    for half_block in block[..64].as_chunks::<32>().0 {
        for shift in 0..4 {
            values.extend(half_block.iter().map(|q| (f32::from(q >> (2 * shift) & 3) - 1.0) * d));
        }
    }
}

/// Base-3 digit `n` of a `TQ1_0` byte, as `ggml` extracts it.
fn digit(byte: u8, n: u32) -> f32 {
    let shifted = u16::from(byte.wrapping_mul(3_u8.pow(n)));
    f32::from((shifted * 3) >> 8) - 1.0
}

/// `TQ1_0`: qs[48] of five base-3 digits, qh[4] of four, half d.
pub(super) fn tq1_0(block: &[u8], values: &mut Vec<f32>) {
    let d = half(&block[52..]);
    for (bytes, digits) in [(&block[..32], 5), (&block[32..48], 5), (&block[48..52], 4)] {
        for n in 0..digits {
            values.extend(bytes.iter().map(|&byte| digit(byte, n) * d));
        }
    }
}

/// `ggml_ue4m3_to_fp32`: an unsigned E4M3 scale, halved for the doubled E2M1
/// table; 0 and 0x7F are zero.
fn ue4m3(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exponent = i32::from(x >> 3 & 0xF);
    let mantissa = f32::from(x & 7);
    let raw = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    raw * 0.5
}

/// `NVFP4`: four UE4M3 scales, one per 16 values, then qs[32] of E2M1
/// indices (values j and j + 8 of a group in one byte).
pub(super) fn nvfp4(block: &[u8], values: &mut Vec<f32>) {
    for group in 0..4 {
        let d = ue4m3(block[group]);
        let qs = &block[4 + 8 * group..12 + 8 * group];
        values.extend(qs.iter().map(|q| f32::from(MXFP4[usize::from(q & 15)]) * d));
        values.extend(qs.iter().map(|q| f32::from(MXFP4[usize::from(q >> 4)]) * d));
    }
}
