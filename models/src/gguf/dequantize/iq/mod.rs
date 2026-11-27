//! llama.cpp v1.0.0 `dequantize_row_iq1_*` .. `_iq3_*`: blocks that index
//! llama.cpp's codebook grids (`grids.txt`, written with the CUDA and Metal
//! copies by `generate.py`).

use std::sync::LazyLock;

use super::half;

/// The grids in `generate.py`'s order: `iq2xxs`, `iq2xs`, `iq2s`, `iq3xxs`,
/// `iq3s`, `iq1s`.
static GRIDS: LazyLock<Vec<Vec<u64>>> = LazyLock::new(|| parse(include_str!("grids.txt")));

/// `name count` headers, each followed by its hexadecimal entries.
fn parse(text: &str) -> Vec<Vec<u64>> {
    let mut tables: Vec<Vec<u64>> = Vec::new();
    for word in text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .flat_map(str::split_whitespace)
    {
        let entry = word.strip_prefix("0x").map(|hex| u64::from_str_radix(hex, 16));
        if word.ends_with("_grid") {
            tables.push(Vec::new());
        } else if let (Some(table), Some(Ok(entry))) = (tables.last_mut(), entry) {
            table.push(entry);
        }
    }
    tables
}

#[derive(Clone, Copy)]
enum Grid {
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
}

/// The bytes of one grid entry; an absent table reads as zeros, which the
/// size check in the tests rules out.
fn entry(grid: Grid, index: usize) -> [u8; 8] {
    GRIDS
        .get(grid as usize)
        .and_then(|table| table.get(index))
        .copied()
        .unwrap_or_default()
        .to_le_bytes()
}

fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn half_word(bytes: &[u8]) -> usize {
    usize::from(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// `ksigns_iq2xs`: seven sign bits and their parity as the eighth.
const fn signs(bits: u32) -> u32 {
    bits | ((bits.count_ones() & 1) << 7)
}

fn signed(scale: f32, magnitudes: &[u8], signs: u32, values: &mut Vec<f32>) {
    values.extend(magnitudes.iter().zip(0..).map(|(&magnitude, bit)| {
        let value = scale * f32::from(magnitude);
        if signs >> bit & 1 == 1 {
            -value
        } else {
            value
        }
    }));
}

fn iq2_scale(d: f32, scale: u8) -> f32 {
    d * (0.5 + f32::from(scale)) * 0.25
}

// Every shifted field below is masked to at most 12 bits.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn iq2_xxs(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    for ib in 0..8 {
        let (indices, aux) = (word(&block[2 + 8 * ib..]), word(&block[6 + 8 * ib..]));
        let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = entry(Grid::Iq2Xxs, ((indices >> (8 * l)) & 0xFF) as usize);
            signed(db, &grid, signs((aux >> (7 * l)) & 127), values);
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn iq2_xs(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    for ib in 0..8 {
        let scales = [iq2_scale(d, block[66 + ib] & 0xF), iq2_scale(d, block[66 + ib] >> 4)];
        for l in 0..4 {
            let value = half_word(&block[2 + 2 * (4 * ib + l)..]);
            signed(
                scales[l / 2],
                &entry(Grid::Iq2Xs, value & 511),
                signs((value >> 9) as u32),
                values,
            );
        }
    }
}

pub(super) fn iq2_s(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    for ib in 0..8 {
        let scales = [iq2_scale(d, block[74 + ib] & 0xF), iq2_scale(d, block[74 + ib] >> 4)];
        for l in 0..4 {
            let high = (usize::from(block[66 + ib]) << (8 - 2 * l)) & 0x300;
            let grid = entry(Grid::Iq2S, usize::from(block[2 + 4 * ib + l]) | high);
            signed(scales[l / 2], &grid, u32::from(block[34 + 4 * ib + l]), values);
        }
    }
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn iq3_xxs(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    for ib in 0..8 {
        let aux = word(&block[66 + 4 * ib..]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let first = entry(Grid::Iq3Xxs, usize::from(block[2 + 8 * ib + 2 * l]));
            let second = entry(Grid::Iq3Xxs, usize::from(block[3 + 8 * ib + 2 * l]));
            let magnitudes = [&first[..4], &second[..4]].concat();
            signed(db, &magnitudes, signs((aux >> (7 * l)) & 127), values);
        }
    }
}

pub(super) fn iq3_s(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    for ib in 0..8 {
        let scale = (block[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        let db = d * f32::from(1 + 2 * scale);
        let high = usize::from(block[66 + ib]);
        for l in 0..4 {
            let first = usize::from(block[2 + 8 * ib + 2 * l]) | ((high << (8 - 2 * l)) & 256);
            let second = usize::from(block[3 + 8 * ib + 2 * l]) | ((high << (7 - 2 * l)) & 256);
            let magnitudes =
                [&entry(Grid::Iq3S, first)[..4], &entry(Grid::Iq3S, second)[..4]].concat();
            signed(db, &magnitudes, u32::from(block[74 + 4 * ib + l]), values);
        }
    }
}

fn iq1_values(scale: f32, index: usize, delta: f32, values: &mut Vec<f32>) {
    let grid = entry(Grid::Iq1S, index);
    values.extend(grid.map(|value| scale * (f32::from(i8::from_le_bytes([value])) + delta)));
}

/// `IQ1_S_DELTA` and `IQ1_M_DELTA`, signed by the block's high bit.
const fn delta(negative: bool) -> f32 {
    if negative {
        -0.125
    } else {
        0.125
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn iq1_s(block: &[u8], values: &mut Vec<f32>) {
    let d = half(block);
    for ib in 0..8 {
        let high = half_word(&block[34 + 2 * ib..]);
        let dl = d * (2 * ((high >> 12) & 7) + 1) as f32;
        for l in 0..4 {
            let index = usize::from(block[2 + 4 * ib + l]) | (((high >> (3 * l)) & 7) << 8);
            iq1_values(dl, index, delta(high & 0x8000 != 0), values);
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn iq1_m(block: &[u8], values: &mut Vec<f32>) {
    let scales: [usize; 4] = std::array::from_fn(|k| half_word(&block[48 + 2 * k..]));
    // The block's half sits in the top nibbles of its four scale words.
    let bits = (scales[0] >> 12)
        | ((scales[1] >> 8) & 0xF0)
        | ((scales[2] >> 4) & 0xF00)
        | (scales[3] & 0xF000);
    let d = half(&(bits as u16).to_le_bytes());
    for ib in 0..8 {
        let packed = scales[ib / 2] >> (6 * (ib % 2));
        let dl = [packed & 7, (packed >> 3) & 7].map(|scale| d * (2 * scale + 1) as f32);
        for l in 0..4 {
            let high = usize::from(block[32 + 2 * ib + l / 2]) >> (4 * (l % 2));
            let index = usize::from(block[4 * ib + l]) | ((high & 7) << 8);
            iq1_values(dl[l / 2], index, delta(high & 8 != 0), values);
        }
    }
}

#[cfg(test)]
mod tests {
    /// Every grid holds as many entries as llama.cpp declares.
    #[test]
    fn grids_hold_llama_cpp_sizes() {
        let sizes: Vec<usize> = super::GRIDS.iter().map(Vec::len).collect();
        assert_eq!(sizes, [256, 512, 1024, 256, 512, 2048]);
    }
}
