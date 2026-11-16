//! Host reference: random GGML blocks with sane scales and a port of
//! llama.cpp v1.0.0 `dequantize_row_q8_0`, `_q4_K` and `_q6_K`.

use mircuda::{bf16, f16};

use super::super::GgmlKernelType;
use crate::Result;

mod iq;
mod k_quants;

use iq::{iq1_m, iq1_s, iq2_s, iq2_xs, iq2_xxs, iq3_s, iq3_xxs, iq4_nl, iq4_xs, mxfp4};
use k_quants::{q2_k, q3_k, q4_k, q5_k, q6_k};

pub(super) struct Blocks {
    pub(super) bytes: Vec<u8>,
}

impl Blocks {
    pub(super) fn random(
        kind: GgmlKernelType,
        rows: usize,
        columns: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut random = Random(seed);
        let mut bytes = random.bytes(kind.bytes(rows, columns)?);
        let block_bytes = usize::try_from(kind.ggml_type().block_bytes())?;
        // Where each block keeps its halves d (and dmin); most start with d.
        let scales: &[usize] = match kind {
            GgmlKernelType::Q4_1
            | GgmlKernelType::Q5_1
            | GgmlKernelType::Q4K
            | GgmlKernelType::Q5K => &[0, 2],
            GgmlKernelType::Q2K => &[80, 82],
            GgmlKernelType::Q3K => &[108],
            GgmlKernelType::Q6K => &[208],
            GgmlKernelType::IQ1M | GgmlKernelType::MXFP4 => &[],
            _ => &[0],
        };
        // Random halves would include infinities and NaNs; keep d and dmin in
        // a range where a 512-wide dot product stays well inside BF16.
        for block in bytes.chunks_exact_mut(block_bytes) {
            for &offset in scales {
                let scale =
                    f16::from_f32(random.unit().mul_add(0.01, 0.001)).to_bits().to_le_bytes();
                block[offset..offset + 2].copy_from_slice(&scale);
            }
            // MXFP4 scales by 2^(e - 128): keep e near 2^-5.
            if kind == GgmlKernelType::MXFP4 {
                block[0] = 118 + random.bytes(1)[0] % 8;
            }
            // IQ1_M has no d field: its half is the top nibbles of scales[0..4].
            if kind == GgmlKernelType::IQ1M {
                let d = f16::from_f32(random.unit().mul_add(0.01, 0.001)).to_bits();
                for (k, at) in (48..56).step_by(2).enumerate() {
                    let nibble = (d >> (4 * k)) & 0xF;
                    let value =
                        (u16::from_le_bytes([block[at], block[at + 1]]) & 0x0FFF) | (nibble << 12);
                    block[at..at + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(Self { bytes })
    }
}

pub(super) fn input(elements: usize, seed: u64) -> Vec<bf16> {
    let mut random = Random(seed);
    (0..elements)
        .map(|_| bf16::from_f32(random.unit().mul_add(2.0, -1.0)))
        .collect()
}

/// The dot product and the sum of absolute terms, which bounds rounding.
pub(super) fn dot(input: &[bf16], weight: &[f64]) -> (f64, f64) {
    input
        .iter()
        .zip(weight)
        .fold((0.0, 0.0), |(total, magnitude), (value, weight)| {
            let term = f64::from(value.to_f32()) * weight;
            (total + term, magnitude + term.abs())
        })
}

/// A host dequantization of one block, appended to the values.
type Dequantize = fn(&[u8], &mut Vec<f64>);

pub(super) fn dequantize(
    kind: GgmlKernelType,
    bytes: &[u8],
    rows: usize,
    columns: usize,
) -> Vec<f64> {
    let (elements, block_bytes, dequantize): (usize, usize, Dequantize) = match kind {
        GgmlKernelType::Q4_0 => (32, 18, |block, values| legacy(block, values, 2, None, false)),
        GgmlKernelType::Q4_1 => (32, 20, |block, values| legacy(block, values, 4, None, true)),
        GgmlKernelType::Q5_0 => (32, 22, |block, values| legacy(block, values, 6, Some(2), false)),
        GgmlKernelType::Q5_1 => (32, 24, |block, values| legacy(block, values, 8, Some(4), true)),
        GgmlKernelType::Q8_0 => (32, 34, q8_0),
        GgmlKernelType::Q2K => (256, 84, q2_k),
        GgmlKernelType::Q3K => (256, 110, q3_k),
        GgmlKernelType::Q4K => (256, 144, q4_k),
        GgmlKernelType::Q5K => (256, 176, q5_k),
        GgmlKernelType::Q6K => (256, 210, q6_k),
        GgmlKernelType::IQ4NL => (32, 18, iq4_nl),
        GgmlKernelType::IQ4XS => (256, 136, iq4_xs),
        GgmlKernelType::IQ2XXS => (256, 66, iq2_xxs),
        GgmlKernelType::IQ2XS => (256, 74, iq2_xs),
        GgmlKernelType::IQ2S => (256, 82, iq2_s),
        GgmlKernelType::IQ3XXS => (256, 98, iq3_xxs),
        GgmlKernelType::IQ3S => (256, 110, iq3_s),
        GgmlKernelType::IQ1S => (256, 50, iq1_s),
        GgmlKernelType::IQ1M => (256, 56, iq1_m),
        GgmlKernelType::MXFP4 => (32, 17, mxfp4),
    };
    let mut values = Vec::with_capacity(rows * columns);
    for block in bytes.chunks_exact(block_bytes).take(rows * columns / elements) {
        dequantize(block, &mut values);
    }
    values
}

fn half(bytes: &[u8]) -> f64 {
    f64::from(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
}

/// `dequantize_row_q4_0`, `_q4_1`, `_q5_0`, `_q5_1`: qs at `qs`, the fifth
/// bits at `qh`, a minimum after d for the `_1` types, else an offset.
fn legacy(block: &[u8], values: &mut Vec<f64>, qs: usize, qh: Option<usize>, minimum: bool) {
    let d = half(block);
    let fifth = qh.map_or(0, |at| {
        u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
    });
    let (offset, minimum) = match (minimum, qh) {
        (true, _) => (0.0, half(&block[2..])),
        (false, Some(_)) => (16.0, 0.0),
        (false, None) => (8.0, 0.0),
    };
    let value = |nibble: u8, bit: u32| {
        let high = if qh.is_some() {
            u8::from(fifth >> bit & 1 == 1) << 4
        } else {
            0
        };
        d.mul_add(f64::from(nibble | high) - offset, minimum)
    };
    let bytes = &block[qs..qs + 16];
    values.extend(bytes.iter().zip(0..).map(|(&q, j)| value(q & 0xF, j)));
    values.extend(bytes.iter().zip(16..).map(|(&q, j)| value(q >> 4, j)));
}

fn q8_0(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    values.extend(block[2..34].iter().map(|&q| d * f64::from(i8::from_le_bytes([q]))));
}

/// `SplitMix64`: deterministic test data without a dependency.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count).map(|_| self.next().to_le_bytes()[0]).collect()
    }

    /// A value in `[0, 1)` from the top 24 bits.
    fn unit(&mut self) -> f32 {
        let bits = u32::try_from(self.next() >> 40).unwrap_or(0);
        f32::from(u16::try_from(bits >> 8).unwrap_or(0)) / 65536.0
    }
}
