use std::path::Path;

use models::{layout::ModelLayout, weights::TensorCatalog};
use runtime::kv::{BlockId, BlockTable, KvWritePlan};
use uuid::Uuid;

use super::*;
use crate::{CudaConfig, Result};

#[test]
fn checkpoint_batched_attention_matches_independent_decode_rows()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let names = names();
    let infos = names.iter().map(|name| required(&catalog, name)).collect::<Result<Vec<_>>>()?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let qkv = qkv(&backend, &tensors, &names)?;
    let weights = weights(&tensors, &names, &qkv)?;
    let config = config();
    let first_table = table(0, 1);
    let second_table = table(1, 3);

    let mut first_scalar = backend.prepare_decode_attention_bf16(config)?;
    let mut second_scalar = backend.prepare_decode_attention_bf16(config)?;
    let mut first_output = backend.inner.pool.allocate(&backend.inner.stream, 2_816)?;
    let mut second_output = backend.inner.pool.allocate(&backend.inner.stream, 2_816)?;
    first_scalar.execute(
        &input(&backend, 0)?,
        weights,
        &KvWritePlan::prefill(Uuid::nil(), 0, &first_table, 0, 1)?,
        &first_table,
        &mut first_output,
    )?;
    second_scalar.execute(
        &input(&backend, 1)?,
        weights,
        &KvWritePlan::prefill(Uuid::nil(), 0, &second_table, 2, 1)?,
        &second_table,
        &mut second_output,
    )?;

    let mut batch = backend.prepare_paged_decode_batch(config.cache, 2, 2)?;
    batch.prepare(&[&first_table, &second_table])?;
    let mut operation = backend.prepare_batched_decode_attention_bf16(config, 2)?;
    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, 5_632)?;
    operation.execute(&input_batch(&backend, 2)?, weights, &batch, &mut actual)?;
    let actual = read(&backend, &actual)?;
    assert_close(&actual[..2_816], &read(&backend, &first_output)?);
    assert_close(&actual[2_816..], &read(&backend, &second_output)?);
    Ok(())
}

fn table(block: u32, tokens: usize) -> BlockTable {
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(block));
    table.set_token_len(tokens);
    table
}
