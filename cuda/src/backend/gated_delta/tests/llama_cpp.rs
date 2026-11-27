//! llama.cpp's GGUF value-head tiling and decay against their definitions.

use mircuda::bf16;

use super::{bf16s, copy, read};
use crate::{
    CudaBackend, CudaConfig, Result,
    kernels::{GatedDeltaLlamaCpp, GatedDeltaTiledValues},
};

const TILED: GatedDeltaTiledValues =
    GatedDeltaTiledValues { key_heads: 2, per_key: 3, head_dim: 4 };

/// Grouped head `k * per_key + j` is stored at `j * key_heads + k`, after
/// the leading rows that stay in place.
#[test]
fn tiled_value_rows_read_each_grouped_head_from_its_stored_head() -> Result<()> {
    let rows = TILED.rows(2, 1)?;
    assert_eq!(rows, [0, 1, 2, 4, 6, 3, 5, 7]);
    let rows = GatedDeltaTiledValues { head_dim: 2, ..TILED }.rows(0, 2)?;
    assert_eq!(&rows[..6], &[0, 1, 4, 5, 8, 9]);
    Ok(())
}

/// Gathering the stored rows into grouped order and tiling the result back
/// returns the stored order; the decay log inverts `-exp`.
#[test]
fn tiling_inverts_the_grouped_gather_on_cuda() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let kernels = GatedDeltaLlamaCpp::compile(&backend.inner.compiler)?;
    let stream = &backend.inner.stream;
    let tokens = 2;
    let width = TILED.key_heads * TILED.per_key * TILED.head_dim;
    #[allow(clippy::cast_precision_loss)]
    let stored = bf16s(&(0..tokens * width).map(|value| value as f32).collect::<Vec<_>>());
    // One token's heads are rows of `head_dim` values.
    let mut indices = Vec::new();
    for token in 0..tokens {
        let heads = u32::try_from(token * TILED.key_heads * TILED.per_key)?;
        indices.extend(TILED.rows(0, 1)?.into_iter().map(|row| row + heads));
    }
    let input = copy(&backend, &stored)?;
    let indices = copy(&backend, &indices)?;
    // A byte view of the grouped values; both share one allocation.
    let grouped = backend.inner.pool.allocate::<bf16>(stream, stored.len())?;
    let row_bytes = 2 * TILED.head_dim;
    let mut grouped_bytes = grouped.reinterpret::<u8>()?;
    kernels.gather_rows(
        stream,
        &input.reinterpret::<u8>()?,
        &mut grouped_bytes,
        &indices,
        row_bytes,
    )?;
    let mut tiled = backend.inner.pool.allocate::<bf16>(stream, stored.len())?;
    kernels.tile_values(stream, TILED, (&grouped, &mut tiled), tokens)?;
    assert_eq!(read(&backend, &tiled)?, stored);
    // The first grouped head of key head 1 is stored head 1.
    let first = read(&backend, &grouped)?;
    assert_eq!(
        first[TILED.per_key * TILED.head_dim..][..TILED.head_dim],
        stored[TILED.head_dim..][..TILED.head_dim]
    );
    let mut decay = copy(&backend, &bf16s(&[-1.0, -std::f32::consts::E]))?;
    kernels.decay_log(stream, &mut decay)?;
    let decay = read(&backend, &decay)?;
    assert!(decay[0].to_f32().abs() < 1.0e-3 && (decay[1].to_f32() - 1.0).abs() < 1.0e-2);
    Ok(())
}
