//! Host reference: random GGML blocks with sane scales and a port of
//! llama.cpp v1.0.0 `dequantize_row_q8_0`, `_q4_K` and `_q6_K`.

use mircuda::{bf16, f16};
use models::gguf::GgmlType;

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
        let scales: &[usize] = match kind.ggml_type() {
            GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
            GgmlType::Q2_K => &[80, 82],
            GgmlType::Q3_K => &[108],
            GgmlType::Q6_K => &[208],
            GgmlType::IQ1_M | GgmlType::MXFP4 | GgmlType::NVFP4 => &[],
            GgmlType::TQ1_0 => &[52],
            GgmlType::TQ2_0 => &[64],
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
            if kind.ggml_type() == GgmlType::MXFP4 {
                block[0] = 118 + random.bytes(1)[0] % 8;
            }
            // NVFP4 scales each 16 values by a UE4M3 byte: exponents 1..4.
            if kind.ggml_type() == GgmlType::NVFP4 {
                for scale in &mut block[..4] {
                    *scale = 8 + random.bytes(1)[0] % 32;
                }
            }
            // IQ1_M has no d field: its half is the top nibbles of
            // scales[0..4].
            if kind.ggml_type() == GgmlType::IQ1_M {
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
    if let Some(values) = shared(kind, bytes, rows, columns) {
        return values;
    }
    let dequantize: Dequantize = match kind.ggml_type() {
        GgmlType::Q4_0 => |block, values| legacy(block, values, 2, None, false),
        GgmlType::Q4_1 => |block, values| legacy(block, values, 4, None, true),
        GgmlType::Q5_0 => |block, values| legacy(block, values, 6, Some(2), false),
        GgmlType::Q5_1 => |block, values| legacy(block, values, 8, Some(4), true),
        GgmlType::Q8_0 => q8_0,
        GgmlType::Q2_K => q2_k,
        GgmlType::Q3_K => q3_k,
        GgmlType::Q4_K => q4_k,
        GgmlType::Q5_K => q5_k,
        GgmlType::Q6_K => q6_k,
        GgmlType::IQ4_NL => iq4_nl,
        GgmlType::IQ4_XS => iq4_xs,
        GgmlType::IQ2_XXS => iq2_xxs,
        GgmlType::IQ2_XS => iq2_xs,
        GgmlType::IQ2_S => iq2_s,
        GgmlType::IQ3_XXS => iq3_xxs,
        GgmlType::IQ3_S => iq3_s,
        GgmlType::IQ1_S => iq1_s,
        GgmlType::IQ1_M => iq1_m,
        GgmlType::MXFP4 => mxfp4,
        other => unreachable!("{other} is decoded by the shared host port"),
    };
    let elements = usize::try_from(kind.ggml_type().block_elements()).unwrap_or(usize::MAX);
    let block_bytes = usize::try_from(kind.ggml_type().block_bytes()).unwrap_or(usize::MAX);
    let mut values = Vec::with_capacity(rows * columns);
    for block in bytes.chunks_exact(block_bytes).take(rows * columns / elements) {
        dequantize(block, &mut values);
    }
    values
}

/// Types without a port here use the `models` host decoder, which is checked
/// against gguf-py.
fn shared(kind: GgmlKernelType, bytes: &[u8], rows: usize, columns: usize) -> Option<Vec<f64>> {
    if !matches!(
        kind.ggml_type(),
        GgmlType::NVFP4 | GgmlType::Q1_0 | GgmlType::Q2_0 | GgmlType::TQ1_0 | GgmlType::TQ2_0
    ) {
        return None;
    }
    let length = kind.bytes(rows, columns).ok()?;
    let values = models::gguf::dequantize(kind.ggml_type(), &bytes[..length], columns).ok()?;
    Some(values.into_iter().map(f64::from).collect())
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
