//! The expanded keys' and values' tails, which the up-projections leave.

use mircuda::bf16;
use models::layout::RotaryPairing;

use super::{
    super::{LatentAttention, LatentAttentionSpec, kernels::ExpandedTails},
    fixture::Runtime,
};
use crate::Result;

/// Values far narrower than the position-free key part (`value_dim <
/// 2 nope_dim - head_dim`) leave a value tail wider than the key tail; every
/// column past `value_dim` must still be zero, and every key column past
/// `nope_dim` the rotary key, then zero. Both buffers start as ones.
#[test]
fn fills_every_column_past_the_projected_ones() -> Result<()> {
    let runtime = Runtime::new()?;
    for (nope_dim, rope_dim, value_dim) in [(112, 16, 64), (128, 64, 128), (64, 32, 16)] {
        let spec = LatentAttentionSpec {
            heads: 3,
            latent_rank: 64,
            rope_dim,
            nope_dim,
            value_dim,
            rotary_pairing: RotaryPairing::Halves,
            block_size: 16,
            max_queries: 1,
            splits: 1,
        };
        let head_dim = 128.max(nope_dim + rope_dim).next_power_of_two();
        let attention = LatentAttention::compile(&runtime.compiler, spec)?;
        let rows = 5;
        let row = spec.row_width();
        let latent = (0..rows * row)
            .map(|index| f32::from(u16::try_from(index % 97).unwrap_or(0)))
            .collect::<Vec<_>>();
        let elements = rows * spec.heads * head_dim;
        let ones = vec![bf16::ONE; elements];
        let (mut keys, mut values) = (runtime.copy(&ones)?, runtime.copy(&ones)?);
        attention.expanded.fill(
            &runtime.stream,
            spec,
            ExpandedTails {
                latent: &runtime.copy_f32(&latent)?,
                keys: &mut keys,
                values: &mut values,
                rows,
                head_dim,
            },
        )?;
        let (keys, values) = (runtime.read(&keys)?, runtime.read(&values)?);
        for head_row in 0..rows * spec.heads {
            let latent_row = head_row / spec.heads;
            for column in 0..head_dim {
                let at = head_row * head_dim + column;
                let key = if column < nope_dim {
                    1.0
                } else if column < nope_dim + rope_dim {
                    latent[latent_row * row + spec.latent_rank + column - nope_dim]
                } else {
                    0.0
                };
                let value = if column < value_dim {
                    1.0
                } else {
                    0.0
                };
                let label =
                    format!("{nope_dim}/{rope_dim}/{value_dim} row {head_row} column {column}");
                assert_eq!(keys[at], bf16::from_f32(key), "key {label}");
                assert_eq!(values[at], bf16::from_f32(value), "value {label}");
            }
        }
    }
    Ok(())
}
