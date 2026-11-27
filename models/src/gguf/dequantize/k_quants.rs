//! `dequantize_row_q2_K` .. `_q6_K`.
//!
//! Products and sums round separately, as llama.cpp computes them.
#![allow(clippy::suboptimal_flops)]

use super::half;

fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (scale, minimum) = if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    };
    (f32::from(scale), f32::from(minimum))
}

pub(super) fn q2_k(block: &[u8], values: &mut Vec<f32>) {
    let (scales, qs) = (&block[..16], &block[16..80]);
    let (d, min) = (half(&block[80..]), half(&block[82..]));
    let mut scale = scales.iter();
    for q in qs.as_chunks::<32>().0 {
        for shift in (0..8).step_by(2) {
            for half_block in [&q[..16], &q[16..]] {
                let sc = scale.next().copied().unwrap_or(0);
                let (dl, ml) = (d * f32::from(sc & 0xF), min * f32::from(sc >> 4));
                values.extend(half_block.iter().map(|q| dl * f32::from((q >> shift) & 3) - ml));
            }
        }
    }
}

pub(super) fn q3_k(block: &[u8], values: &mut Vec<f32>) {
    let (hmask, qs, packed) = (&block[..32], &block[32..96], &block[96..108]);
    let d = half(&block[108..]);
    let scale = |j: usize| {
        let low = if j < 8 {
            packed[j] & 15
        } else {
            packed[j - 8] >> 4
        };
        let high = (packed[8 + j % 4] >> (2 * (j / 4))) & 3;
        f32::from(i16::from(low | (high << 4)) - 32)
    };
    let mut is = 0;
    for (part, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        for (step, shift) in (0..8).step_by(2).enumerate() {
            let bit = 4 * part + step;
            for range in [0..16, 16..32] {
                let dl = d * scale(is);
                is += 1;
                values.extend(range.map(|l| {
                    let low = i16::from((q[l] >> shift) & 3);
                    let high = if (hmask[l] >> bit) & 1 == 1 {
                        0
                    } else {
                        4
                    };
                    dl * f32::from(low - high)
                }));
            }
        }
    }
}

pub(super) fn q4_k(block: &[u8], values: &mut Vec<f32>) {
    k4_k5(block, values, None);
}

pub(super) fn q5_k(block: &[u8], values: &mut Vec<f32>) {
    k4_k5(block, values, Some(&block[16..48]));
}

/// `Q4_K` and `Q5_K`: half d, half dmin, scales[12], [qh[32]], qs[128].
fn k4_k5(block: &[u8], values: &mut Vec<f32>, qh: Option<&[u8]>) {
    let (d, min, scales) = (half(block), half(&block[2..]), &block[4..16]);
    let qs = &block[if qh.is_some() {
        48
    } else {
        16
    }..];
    for (j, q) in qs.as_chunks::<32>().0.iter().take(4).enumerate() {
        for (part, shift) in [0, 4].into_iter().enumerate() {
            let s = 2 * j + part;
            let (sc, m) = scale_min_k4(s, scales);
            values.extend(q.iter().enumerate().map(|(l, byte)| {
                let high = qh.map_or(0, |qh| ((qh[l] >> s) & 1) << 4);
                d * sc * f32::from(((byte >> shift) & 0xF) | high) - min * m
            }));
        }
    }
}

pub(super) fn q6_k(block: &[u8], values: &mut Vec<f32>) {
    let d = half(&block[208..]);
    for part in 0..2 {
        let ql = &block[64 * part..64 * part + 64];
        let qh = &block[128 + 32 * part..160 + 32 * part];
        let sc = &block[192 + 8 * part..200 + 8 * part];
        for quarter in 0..4 {
            let (low, low_shift) = (
                &ql[32 * (quarter & 1)..],
                if quarter < 2 {
                    0
                } else {
                    4
                },
            );
            values.extend((0..32).map(|l| {
                let q = ((low[l] >> low_shift) & 0xF) | (((qh[l] >> (2 * quarter)) & 3) << 4);
                let scale = f32::from(i8::from_le_bytes([sc[l / 16 + 2 * quarter]]));
                d * scale * f32::from(i16::from(q) - 32)
            }));
        }
    }
}
