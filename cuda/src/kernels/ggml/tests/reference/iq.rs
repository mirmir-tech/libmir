//! Host ports of llama.cpp v1.0.0 `dequantize_row_iq*`.

use super::half;

/// `kvalues_iq4nl` from `ggml-common.h`.
const IQ4_VALUES: [i8; 16] =
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

fn iq4_values(qs: &[u8], scale: f64, values: &mut Vec<f64>) {
    let value = |index: u8| scale * f64::from(IQ4_VALUES[usize::from(index)]);
    values.extend(qs.iter().map(|&q| value(q & 0xF)));
    values.extend(qs.iter().map(|&q| value(q >> 4)));
}

/// `kvalues_mxfp4`: e2m1 values, doubled.
const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `dequantize_row_mxfp4`: `GGML_E8M0_TO_FP32_HALF(e)` times the doubled table.
pub(super) fn mxfp4(block: &[u8], values: &mut Vec<f64>) {
    let scale = 2.0_f64.powi(i32::from(block[0]) - 128);
    let value = |index: u8| scale * f64::from(MXFP4_VALUES[usize::from(index)]);
    values.extend(block[1..17].iter().map(|&q| value(q & 0xF)));
    values.extend(block[1..17].iter().map(|&q| value(q >> 4)));
}

pub(super) fn iq4_nl(block: &[u8], values: &mut Vec<f64>) {
    iq4_values(&block[2..18], half(block), values);
}

pub(super) fn iq4_xs(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    let high = u16::from_le_bytes([block[2], block[3]]);
    for (sub, qs) in block[8..136].as_chunks::<16>().0.iter().enumerate() {
        let low = (block[4 + sub / 2] >> (4 * (sub % 2))) & 0xF;
        let upper = u8::try_from((high >> (2 * sub)) & 3).unwrap_or(0);
        let scale = i32::from(low | (upper << 4)) - 32;
        iq4_values(qs, d * f64::from(scale), values);
    }
}

/// A grid table parsed from the generated CUDA source, one source of truth.
fn grid(source: &str, name: &str) -> Vec<u64> {
    let start = source.find(&format!("k_{name}[")).unwrap_or(0);
    let body = &source[start..source[start..].find("};").map_or(source.len(), |end| start + end)];
    body.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|word| word.strip_prefix("0x"))
        .filter_map(|hex| u64::from_str_radix(hex.trim_end_matches(['u', 'l']), 16).ok())
        .collect()
}

static IQ2: &str = include_str!("../../../../../kernels/ggml/grids/iq2.cuh");
static IQ13: &str = include_str!("../../../../../kernels/ggml/grids/iq13.cuh");
static GRIDS: std::sync::LazyLock<[Vec<u64>; 6]> = std::sync::LazyLock::new(|| {
    [
        grid(IQ2, "iq2xxs_grid"),
        grid(IQ2, "iq2xs_grid"),
        grid(IQ2, "iq2s_grid"),
        grid(IQ13, "iq3xxs_grid"),
        grid(IQ13, "iq3s_grid"),
        grid(IQ13, "iq1s_grid"),
    ]
});

fn entry(table: usize, index: usize) -> [u8; 8] {
    GRIDS[table].get(index).copied().unwrap_or(0).to_le_bytes()
}

fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn half_word(bytes: &[u8]) -> usize {
    usize::from(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// `ksigns_iq2xs`: seven sign bits and their parity as the eighth.
fn signs(bits: u32) -> u32 {
    bits | ((bits.count_ones() & 1) << 7)
}

fn signed(scale: f64, magnitudes: &[u8], signs: u32, values: &mut Vec<f64>) {
    values.extend(magnitudes.iter().zip(0..).map(|(&m, j)| {
        let sign = if signs >> j & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        scale * f64::from(m) * sign
    }));
}

fn iq2_scale(d: f64, scale: u8) -> f64 {
    d * (0.5 + f64::from(scale)) * 0.25
}

pub(super) fn iq2_xxs(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    for ib in 0..8 {
        let (indices, aux) = (word(&block[2 + 8 * ib..]), word(&block[6 + 8 * ib..]));
        let db = d * (0.5 + f64::from(aux >> 28)) * 0.25;
        for l in 0..4 {
            let grid = entry(0, usize::try_from((indices >> (8 * l)) & 0xFF).unwrap_or(0));
            signed(db, &grid, signs((aux >> (7 * l)) & 127), values);
        }
    }
}

pub(super) fn iq2_xs(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    for ib in 0..8 {
        let scales = [iq2_scale(d, block[66 + ib] & 0xF), iq2_scale(d, block[66 + ib] >> 4)];
        for l in 0..4 {
            let value = half_word(&block[2 + 2 * (4 * ib + l)..]);
            let bits = u32::try_from(value >> 9).unwrap_or(0);
            signed(scales[l / 2], &entry(1, value & 511), signs(bits), values);
        }
    }
}

pub(super) fn iq2_s(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    for ib in 0..8 {
        let scales = [iq2_scale(d, block[74 + ib] & 0xF), iq2_scale(d, block[74 + ib] >> 4)];
        for l in 0..4 {
            let high = (usize::from(block[66 + ib]) << (8 - 2 * l)) & 0x300;
            let grid = entry(2, usize::from(block[2 + 4 * ib + l]) | high);
            signed(scales[l / 2], &grid, u32::from(block[34 + 4 * ib + l]), values);
        }
    }
}

pub(super) fn iq3_xxs(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    for ib in 0..8 {
        let aux = word(&block[66 + 4 * ib..]);
        let db = d * (0.5 + f64::from(aux >> 28)) * 0.5;
        for l in 0..4 {
            let first = entry(3, usize::from(block[2 + 8 * ib + 2 * l]));
            let second = entry(3, usize::from(block[3 + 8 * ib + 2 * l]));
            let magnitudes = [&first[..4], &second[..4]].concat();
            signed(db, &magnitudes, signs((aux >> (7 * l)) & 127), values);
        }
    }
}

pub(super) fn iq3_s(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    for ib in 0..8 {
        let scale = (block[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        let db = d * f64::from(1 + 2 * scale);
        let high = usize::from(block[66 + ib]);
        for l in 0..4 {
            let first = usize::from(block[2 + 8 * ib + 2 * l]) | ((high << (8 - 2 * l)) & 256);
            let second = usize::from(block[3 + 8 * ib + 2 * l]) | ((high << (7 - 2 * l)) & 256);
            let magnitudes = [&entry(4, first)[..4], &entry(4, second)[..4]].concat();
            signed(db, &magnitudes, u32::from(block[74 + 4 * ib + l]), values);
        }
    }
}

fn iq1_values(scale: f64, index: usize, delta: f64, values: &mut Vec<f64>) {
    values.extend(entry(5, index).map(|g| scale * (f64::from(i8::from_le_bytes([g])) + delta)));
}

pub(super) fn iq1_s(block: &[u8], values: &mut Vec<f64>) {
    let d = half(block);
    for ib in 0..8 {
        let high = half_word(&block[34 + 2 * ib..]);
        let dl = d * f64::from(u32::try_from(2 * ((high >> 12) & 7) + 1).unwrap_or(0));
        let delta = if high & 0x8000 == 0 {
            0.125
        } else {
            -0.125
        };
        for l in 0..4 {
            let index = usize::from(block[2 + 4 * ib + l]) | (((high >> (3 * l)) & 7) << 8);
            iq1_values(dl, index, delta, values);
        }
    }
}

pub(super) fn iq1_m(block: &[u8], values: &mut Vec<f64>) {
    let scales: Vec<usize> = (0..4).map(|k| half_word(&block[48 + 2 * k..])).collect();
    let bits = (scales[0] >> 12)
        | ((scales[1] >> 8) & 0xF0)
        | ((scales[2] >> 4) & 0xF00)
        | (scales[3] & 0xF000);
    let d = f64::from(mircuda::f16::from_bits(u16::try_from(bits).unwrap_or(0)).to_f32());
    for ib in 0..8 {
        let packed = scales[ib / 2] >> (6 * (ib % 2));
        let dl = [packed & 7, (packed >> 3) & 7]
            .map(|s| d * f64::from(u32::try_from(2 * s + 1).unwrap_or(0)));
        for l in 0..4 {
            let high = usize::from(block[32 + 2 * ib + l / 2]) >> (4 * (l % 2));
            let index = usize::from(block[4 * ib + l]) | ((high & 7) << 8);
            let delta = if high & 8 == 0 {
                0.125
            } else {
                -0.125
            };
            iq1_values(dl[l / 2], index, delta, values);
        }
    }
}
