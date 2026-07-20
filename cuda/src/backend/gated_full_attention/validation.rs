use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{AffineGatedFullAttentionConfig, CudaAffineGatedFullAttentionState, checked};
use crate::{Error, Result};

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_execution(
    config: AffineGatedFullAttentionConfig,
    tokens: usize,
    input: &DeviceBuffer<bf16>,
    positions: &DeviceBuffer<u32>,
    state: &CudaAffineGatedFullAttentionState,
    write_plan: &KvWritePlan,
    table: &BlockTable,
    start_position: usize,
    output: &DeviceBuffer<bf16>,
) -> Result<()> {
    exact("gated attention input", checked(tokens, config.hidden_size)?, input.len())?;
    exact("gated attention positions", checked(tokens, 3)?, positions.len())?;
    exact("gated attention output", checked(tokens, config.hidden_size)?, output.len())?;
    let storage = state.storage_spec();
    if storage.kv_heads != config.key_value_heads
        || storage.key_head_dim != config.head_dim
        || storage.value_head_dim != config.head_dim
        || write_plan.token_count() != tokens
        || table.token_len() != start_position.saturating_add(tokens)
    {
        return Err(Error::InvalidPagedKv("gated attention execution metadata mismatch"));
    }
    Ok(())
}

fn exact(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(Error::InvalidTensorSize { name: name.into(), expected, actual });
    }
    Ok(())
}
