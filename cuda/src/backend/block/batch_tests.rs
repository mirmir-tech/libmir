use std::path::Path;

use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::kv::{BlockId, BlockTable, KvWritePlan};
use uuid::Uuid;

use super::{CudaBackend, tests::*};
use crate::{CudaConfig, CudaMoeBatchPolicy, CudaPlanningPolicy};

#[test]
fn checkpoint_batched_moe_block_matches_independent_decode_rows()
-> Result<(), Box<dyn std::error::Error>> {
    compare(CudaConfig::default())
}

#[test]
fn checkpoint_weight_only_batch_matches_independent_decode_rows()
-> Result<(), Box<dyn std::error::Error>> {
    compare(CudaConfig {
        planning: CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A16,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })
}

#[test]
fn checkpoint_bucketed_w4a4_batch_matches_independent_decode_rows()
-> Result<(), Box<dyn std::error::Error>> {
    compare(CudaConfig {
        planning: CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A4Bucketed,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })
}

fn compare(config: CudaConfig) -> Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(config)?;
    let load = load_config();
    let template = backend.load_nvfp4_moe_layer_template(&decoder, &catalog, LAYER, load)?;
    let first_table = table(0, 1);
    let second_table = table(1, 3);

    let first_input = input(&backend, 0)?;
    let first_output = backend.inner.pool.allocate(&backend.inner.stream, 2_816)?;
    let mut first = template.instantiate(&first_input, &first_output)?;
    first.execute(&KvWritePlan::prefill(Uuid::nil(), LAYER, &first_table, 0, 1)?, &first_table)?;
    let second_input = input(&backend, 1)?;
    let second_output = backend.inner.pool.allocate(&backend.inner.stream, 2_816)?;
    let mut second = template.instantiate(&second_input, &second_output)?;
    second
        .execute(&KvWritePlan::prefill(Uuid::nil(), LAYER, &second_table, 2, 1)?, &second_table)?;

    let batch_input = copy(&backend, &[input_values(0)?, input_values(1)?].concat())?;
    let mut metadata =
        backend.prepare_paged_decode_batch(template.config().attention.cache, 2, 2)?;
    metadata.prepare(&[&first_table, &second_table])?;
    let mut operation = template.instantiate_batch(2)?;
    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, 5_632)?;
    operation.execute(&batch_input, &metadata, &mut actual)?;
    let actual = read(&backend, &actual)?;
    close(&actual[..2_816], &read(&backend, &first_output)?);
    close(&actual[2_816..], &read(&backend, &second_output)?);
    Ok(())
}

fn table(block: u32, tokens: usize) -> BlockTable {
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(block));
    table.set_token_len(tokens);
    table
}
