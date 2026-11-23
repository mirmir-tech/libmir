//! Host port of llama.cpp v1.0.0 `dequantize_row_iq1_*` .. `_iq3_*`, reading
//! the grid tables the Metal decoders compile in.

use std::sync::LazyLock;

use models::gguf::GgmlType;

static GRIDS: LazyLock<Vec<(String, Vec<u64>)>> = LazyLock::new(|| {
    let source = include_str!("../../../../../kernels/ggml/grids.h");
    source
        .split("constant ")
        .skip(1)
        .map(|table| {
            let name = table.split(['[', ' ']).nth(1).unwrap_or_default().to_owned();
            let values = table
                .split("0x")
                .skip(1)
                .map(|value| {
                    let end = value.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(value.len());
                    u64::from_str_radix(&value[..end], 16)
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_default();
            (name, values)
        })
        .collect()
});

fn grid(name: &str, index: usize) -> [u8; 8] {
    let table = GRIDS.iter().find(|(table, _)| table == name).map(|(_, values)| values);
    table
        .and_then(|values| values.get(index))
        .copied()
        .unwrap_or_default()
        .to_le_bytes()
}

fn half(bytes: &[u8]) -> f32 {
    models::gguf::dequantize(GgmlType::F16, &bytes[..2], 1).map_or(f32::NAN, |values| values[0])
}

fn u16_at(block: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([block[at], block[at + 1]]))
}

fn u32_at(block: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
}

/// `ksigns_iq2xs`: seven bits and their parity.
fn even_signs(bits: u32) -> u32 {
    bits | ((bits.count_ones() & 1) << 7)
}

fn signed(magnitude: u8, signs: u32, j: usize) -> f32 {
    let value = f32::from(magnitude);
    if (signs >> j) & 1 == 1 {
        -value
    } else {
        value
    }
}

/// A small unsigned integer as `f32`.
fn small(value: impl TryInto<u16>) -> f32 {
    f32::from(value.try_into().unwrap_or_default())
}

/// Values of one IQ1-IQ3 block of 256.
pub(super) fn dequantize(ggml_type: GgmlType, block: &[u8]) -> Vec<f32> {
    let mut values = Vec::with_capacity(256);
    for ib in 0..8 {
        match ggml_type {
            GgmlType::IQ2_XXS => iq2_xxs(block, ib, &mut values),
            GgmlType::IQ2_XS | GgmlType::IQ2_S => iq2_xs_s(ggml_type, block, ib, &mut values),
            GgmlType::IQ3_XXS => iq3_xxs(block, ib, &mut values),
            GgmlType::IQ3_S => iq3_s(block, ib, &mut values),
            GgmlType::IQ1_S => iq1_s(block, ib, &mut values),
            GgmlType::IQ1_M => iq1_m(block, ib, &mut values),
            _ => {},
        }
    }
    values
}

fn iq2_xxs(block: &[u8], ib: usize, values: &mut Vec<f32>) {
    let (indices, aux) = (u32_at(block, 2 + 8 * ib), u32_at(block, 6 + 8 * ib));
    let db = half(block) * (0.5 + small(aux >> 28)) * 0.25;
    for l in 0..4 {
        let grid = grid("k_iq2xxs_grid", usize::try_from((indices >> (8 * l)) & 0xFF).unwrap_or(0));
        let signs = even_signs((aux >> (7 * l)) & 127);
        values.extend((0..8).map(|j| db * signed(grid[j], signs, j)));
    }
}

fn iq2_xs_s(ggml_type: GgmlType, block: &[u8], ib: usize, values: &mut Vec<f32>) {
    let scales = block[if ggml_type == GgmlType::IQ2_XS {
        66
    } else {
        74
    } + ib];
    let d = half(block);
    for l in 0..4 {
        let db = d * (0.5 + f32::from((scales >> (4 * (l / 2))) & 15)) * 0.25;
        let (grid, signs) = if ggml_type == GgmlType::IQ2_XS {
            let value = u16_at(block, 2 + 2 * (4 * ib + l));
            let signs = even_signs(u32::try_from(value >> 9).unwrap_or(0));
            (grid("k_iq2xs_grid", value & 511), signs)
        } else {
            let high = usize::from(block[66 + ib]) << (8 - 2 * l) & 0x300;
            let index = usize::from(block[2 + 4 * ib + l]) | high;
            (grid("k_iq2s_grid", index), u32::from(block[34 + 4 * ib + l]))
        };
        values.extend((0..8).map(|j| db * signed(grid[j], signs, j)));
    }
}

fn iq3_xxs(block: &[u8], ib: usize, values: &mut Vec<f32>) {
    let aux = u32_at(block, 66 + 4 * ib);
    let db = half(block) * (0.5 + small(aux >> 28)) * 0.5;
    for l in 0..4 {
        let signs = even_signs((aux >> (7 * l)) & 127);
        let first = grid("k_iq3xxs_grid", usize::from(block[2 + 8 * ib + 2 * l]));
        let second = grid("k_iq3xxs_grid", usize::from(block[3 + 8 * ib + 2 * l]));
        let magnitudes = [&first[..4], &second[..4]].concat();
        values.extend((0..8).map(|j| db * signed(magnitudes[j], signs, j)));
    }
}

fn iq3_s(block: &[u8], ib: usize, values: &mut Vec<f32>) {
    let scale = (block[106 + ib / 2] >> (4 * (ib % 2))) & 15;
    let db = half(block) * f32::from(1 + 2 * scale);
    let high = usize::from(block[66 + ib]);
    for l in 0..4 {
        let signs = u32::from(block[74 + 4 * ib + l]);
        let first = usize::from(block[2 + 8 * ib + 2 * l]) | (high << (8 - 2 * l) & 256);
        let second = usize::from(block[3 + 8 * ib + 2 * l]) | (high << (7 - 2 * l) & 256);
        let magnitudes =
            [&grid("k_iq3s_grid", first)[..4], &grid("k_iq3s_grid", second)[..4]].concat();
        values.extend((0..8).map(|j| db * signed(magnitudes[j], signs, j)));
    }
}

fn iq1(grid: [u8; 8], dl: f32, negative: bool, values: &mut Vec<f32>) {
    let delta = if negative {
        -0.125
    } else {
        0.125
    };
    values.extend(grid.iter().map(|&g| dl * (f32::from(i8::from_le_bytes([g])) + delta)));
}

fn iq1_s(block: &[u8], ib: usize, values: &mut Vec<f32>) {
    let qh = u16_at(block, 34 + 2 * ib);
    let dl = half(block) * small(2 * ((qh >> 12) & 7) + 1);
    for l in 0..4 {
        let index = usize::from(block[2 + 4 * ib + l]) | ((qh >> (3 * l)) & 7) << 8;
        iq1(grid("k_iq1s_grid", index), dl, qh & 0x8000 != 0, values);
    }
}

fn iq1_m(block: &[u8], ib: usize, values: &mut Vec<f32>) {
    let sc = [48, 50, 52, 54].map(|at| u16_at(block, at));
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    let d = half(&u16::try_from(bits).unwrap_or(0).to_le_bytes());
    for l in 0..4 {
        let packed = (sc[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7;
        let high = usize::from(block[32 + 2 * ib + l / 2] >> (4 * (l % 2)));
        let index = usize::from(block[4 * ib + l]) | (high & 7) << 8;
        iq1(grid("k_iq1s_grid", index), d * small(2 * packed + 1), high & 8 != 0, values);
    }
}
