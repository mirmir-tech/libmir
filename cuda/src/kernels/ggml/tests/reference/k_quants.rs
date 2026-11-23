//! Host ports of llama.cpp v1.0.0 `dequantize_row_q2_K` through `_q6_K`.

use super::half;

/// `dequantize_row_q3_K`: 2-bit values from qs, a third bit from hmask
/// (clear means minus 4), sixteen 6-bit scales offset by 32.
pub(super) fn q3_k(block: &[u8], values: &mut Vec<f64>) {
    let d = half(&block[108..]);
    let (mask, qs) = (&block[0..32], &block[32..96]);
    let word =
        |at: usize| u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
    let (raw0, raw1, raw2) = (word(96), word(100), word(104));
    let (low, high) = (0x0F0F_0F0F, 0x0303_0303);
    let scales = [
        (raw0 & low) | ((raw2 & high) << 4),
        (raw1 & low) | (((raw2 >> 2) & high) << 4),
        ((raw0 >> 4) & low) | (((raw2 >> 4) & high) << 4),
        ((raw1 >> 4) & low) | (((raw2 >> 6) & high) << 4),
    ]
    .map(u32::to_le_bytes)
    .concat();
    let mut index = 0;
    let mut bit = 1_u8;
    for q in qs.as_chunks::<32>().0 {
        for shift in [0, 2, 4, 6] {
            for range in [0..16, 16..32] {
                let dl = d * f64::from(i32::from(scales[index]) - 32);
                index += 1;
                values.extend(range.map(|l| {
                    let minus = if mask[l] & bit == 0 {
                        4
                    } else {
                        0
                    };
                    dl * f64::from(i32::from((q[l] >> shift) & 3) - minus)
                }));
            }
            bit <<= 1;
        }
    }
}

/// `dequantize_row_q2_K`: 2-bit values, a 4-bit scale and minimum per 16.
pub(super) fn q2_k(block: &[u8], values: &mut Vec<f64>) {
    let (d, minimum) = (half(&block[80..]), half(&block[82..]));
    let mut scales = block[0..16].iter();
    for q in block[16..80].as_chunks::<32>().0 {
        for shift in [0, 2, 4, 6] {
            for range in [0..16, 16..32] {
                let packed = scales.next().copied().unwrap_or(0);
                let (dl, ml) = (d * f64::from(packed & 0xF), minimum * f64::from(packed >> 4));
                values.extend(range.map(|l| dl.mul_add(f64::from((q[l] >> shift) & 3), -ml)));
            }
        }
    }
}

pub(super) fn q5_k(block: &[u8], values: &mut Vec<f64>) {
    let (d, minimum) = (half(block), half(&block[2..]));
    let (scales, qh) = (&block[4..16], &block[16..48]);
    for (pair, q) in block[48..176].as_chunks::<32>().0.iter().enumerate() {
        for (part, shift) in [(0, 0), (1, 4)] {
            let s = 2 * pair + part;
            let (scale, offset) = scale_min_k4(s, scales);
            values.extend(q.iter().zip(qh).map(|(&q, &h)| {
                let value = (q >> shift) & 0xF | ((h >> s) & 1) << 4;
                minimum.mul_add(-offset, d * scale * f64::from(value))
            }));
        }
    }
}

fn scale_min_k4(j: usize, q: &[u8]) -> (f64, f64) {
    if j < 4 {
        (f64::from(q[j] & 63), f64::from(q[j + 4] & 63))
    } else {
        (
            f64::from((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
            f64::from((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
        )
    }
}

pub(super) fn q4_k(block: &[u8], values: &mut Vec<f64>) {
    let (d, minimum) = (half(block), half(&block[2..]));
    let scales = &block[4..16];
    for (pair, q) in block[16..144].as_chunks::<32>().0.iter().enumerate() {
        let (scale, offset) = scale_min_k4(2 * pair, scales);
        values.extend(q.iter().map(|&q| minimum.mul_add(-offset, d * scale * f64::from(q & 0xF))));
        let (scale, offset) = scale_min_k4(2 * pair + 1, scales);
        values.extend(q.iter().map(|&q| minimum.mul_add(-offset, d * scale * f64::from(q >> 4))));
    }
}

pub(super) fn q6_k(block: &[u8], values: &mut Vec<f64>) {
    let d = half(&block[208..]);
    for half_block in 0..2 {
        let ql = &block[half_block * 64..half_block * 64 + 64];
        let qh = &block[128 + half_block * 32..128 + half_block * 32 + 32];
        let scales = &block[192 + half_block * 8..192 + half_block * 8 + 8];
        let mut out = [0.0; 128];
        for l in 0..32 {
            let scale = |index: usize| f64::from(i8::from_le_bytes([scales[l / 16 + index]]));
            let value = |low: u8, shift: u32| {
                f64::from(i16::from(low | (((qh[l] >> shift) & 3) << 4)) - 32)
            };
            out[l] = d * scale(0) * value(ql[l] & 0xF, 0);
            out[l + 32] = d * scale(2) * value(ql[l + 32] & 0xF, 2);
            out[l + 64] = d * scale(4) * value(ql[l] >> 4, 4);
            out[l + 96] = d * scale(6) * value(ql[l + 32] >> 4, 6);
        }
        values.extend(out);
    }
}
