//! Host reference: random GGML blocks with sane scales, decoded by the
//! `models` port of llama.cpp v1.0.0 `dequantize_row_*`.

use mircuda::{bf16, f16};
use models::gguf::GgmlType;

use super::super::GgmlKernelType;
use crate::Result;

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

/// The `models` host decoding of `rows` rows of `columns` values; the
/// blocks past them stay unread.
pub(super) fn dequantize(
    kind: GgmlKernelType,
    bytes: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f64>> {
    let length = kind.bytes(rows, columns)?;
    let values = models::gguf::dequantize(kind.ggml_type(), &bytes[..length], columns)?;
    Ok(values.into_iter().map(f64::from).collect())
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
