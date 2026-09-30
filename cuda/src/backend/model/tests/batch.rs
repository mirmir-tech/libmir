use std::path::Path;

use models::{layout::DecoderConfig, weights::TensorCatalog};
use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable},
};
use uuid::Uuid;

use super::{assertions::assert_logits_close, projection_gate, prompt, read, template};
use crate::{
    CudaBackend, CudaConfig, CudaModelSessionConfig, CudaOutputHeadPolicy, CudaPlanningPolicy,
};

#[test]
fn checkpoint_model_batch_matches_independent_scalar_rows() -> Result<(), Box<dyn std::error::Error>>
{
    let (root, dense) = if let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL") {
        (root, true)
    } else if let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") {
        (root, false)
    } else {
        return Ok(());
    };
    let layout = models::layout::ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(CudaConfig {
        planning: CudaPlanningPolicy {
            output_head: CudaOutputHeadPolicy::Bf16,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })?;
    let template = if dense {
        projection_gate::load_template(&backend, &decoder, &catalog)?
    } else {
        template(&backend, &decoder, &catalog)?
    };
    let rows = prompt::rows(&layout)?;
    let sessions = [Uuid::new_v4(), Uuid::new_v4()];
    let blocks = [BlockId(0), BlockId(1)];
    let mut reference = template.instantiate()?;
    let mut expected = Vec::with_capacity(2);
    for ((session, block), tokens) in sessions.iter().zip(blocks).zip(&rows) {
        let mut logits = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            logits = read(&backend, reference.decode(*session, *token, &table(block, index + 1))?)?;
        }
        expected.push(logits);
    }
    let (first, second) = (&expected[0], &expected[1]);

    let caches = template.allocate_shared_kv()?;
    let mut scalar =
        template.instantiate_with_config_and_caches(CudaModelSessionConfig::default(), &caches)?;
    for ((session, block), tokens) in sessions.iter().zip(blocks).zip(&rows) {
        for (index, token) in tokens[..tokens.len() - 1].iter().enumerate() {
            scalar.decode(*session, *token, &table(block, index + 1))?;
        }
    }
    let tables = [table(blocks[0], rows[0].len()), table(blocks[1], rows[1].len())];
    let last = [rows[0][rows[0].len() - 1], rows[1][rows[1].len() - 1]];
    let mut batch = template.instantiate_decode_batch_with_caches(2, &caches)?;
    let logits = read(&backend, batch.decode(&last, &[&tables[0], &tables[1]])?)?;
    let (actual_first, actual_second) = logits.split_at(decoder.vocab_size);
    assert_logits_close(actual_first, first, 0.1);
    assert_logits_close(actual_second, second, 0.1);
    let scalar_tokens = [u32::try_from(maximum(first))?, u32::try_from(maximum(second))?];
    let batch_tokens =
        [u32::try_from(maximum(actual_first))?, u32::try_from(maximum(actual_second))?];
    if !dense {
        assert_eq!(batch_tokens, scalar_tokens);
    }
    let selected = batch.sample(&[SamplingLogits::None, SamplingLogits::None])?;
    let mut host = backend.inner.context.allocate_pinned(2)?;
    backend.inner.stream.copy_to_host(selected, &mut host)?;
    let selected = host.to_vec()?;
    for (row, (selected, expected)) in selected.into_iter().zip(batch_tokens).enumerate() {
        let values = if row == 0 {
            actual_first
        } else {
            actual_second
        };
        assert_eq!(
            selected,
            expected,
            "batch sampler selected score {} instead of {}",
            values[selected as usize].to_f32(),
            values[expected as usize].to_f32(),
        );
    }
    Ok(())
}

#[test]
fn checkpoint_dense_batch_preserves_real_prompt_tokens() -> Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL") else {
        return Ok(());
    };
    let layout = models::layout::ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = exact_backend()?;
    let template = projection_gate::load_template(&backend, &decoder, &catalog)?;
    let prompts = projection_gate::prompts(&layout)?;
    let sessions = [Uuid::new_v4(), Uuid::new_v4()];
    let mut tables = [prompt_table(prompts[0].len(), 0)?, prompt_table(prompts[1].len(), 16)?];
    let caches = template.allocate_shared_kv()?;
    let mut scalar =
        template.instantiate_with_config_and_caches(CudaModelSessionConfig::default(), &caches)?;
    let mut tokens = [0_u32; 2];
    for row in 0..2 {
        scalar.prefill_from(sessions[row], &prompts[row], 0, &tables[row])?;
        tokens[row] = read_token(&backend, scalar.sample(SamplingLogits::None)?)?;
        tables[row].set_token_len(prompts[row].len() + 1);
    }
    let first = read(&backend, scalar.decode(sessions[0], tokens[0], &tables[0])?)?;
    let second = read(&backend, scalar.decode(sessions[1], tokens[1], &tables[1])?)?;
    let mut batch = template.instantiate_decode_batch_with_caches(2, &caches)?;
    let logits = read(&backend, batch.decode(&tokens, &[&tables[0], &tables[1]])?)?;
    assert_rows(&logits, &first, &second, decoder.vocab_size);

    tokens = [u32::try_from(maximum(&first))?, u32::try_from(maximum(&second))?];
    tables[0].set_token_len(prompts[0].len() + 2);
    tables[1].set_token_len(prompts[1].len() + 2);
    let first = read(&backend, scalar.decode(sessions[0], tokens[0], &tables[0])?)?;
    let second = read(&backend, scalar.decode(sessions[1], tokens[1], &tables[1])?)?;
    let replay = read(&backend, batch.decode(&tokens, &[&tables[0], &tables[1]])?)?;
    assert_rows(&replay, &first, &second, decoder.vocab_size);
    Ok(())
}

fn assert_rows(
    actual: &[mircuda::bf16],
    first: &[mircuda::bf16],
    second: &[mircuda::bf16],
    vocab: usize,
) {
    let (actual_first, actual_second) = actual.split_at(vocab);
    assert_logits_close(actual_first, first, 0.1);
    assert_logits_close(actual_second, second, 0.1);
    assert_eq!(maximum(actual_first), maximum(first));
    assert_eq!(maximum(actual_second), maximum(second));
}

fn exact_backend() -> crate::Result<CudaBackend> {
    CudaBackend::new(CudaConfig {
        planning: CudaPlanningPolicy {
            output_head: CudaOutputHeadPolicy::Bf16,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })
}

fn prompt_table(tokens: usize, first_block: u32) -> crate::Result<BlockTable> {
    let mut table = BlockTable::with_block_size(16);
    for offset in 0..tokens.div_ceil(16) {
        table.push(BlockId(first_block + u32::try_from(offset)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}

fn read_token(backend: &CudaBackend, token: &mircuda::DeviceBuffer<u32>) -> crate::Result<u32> {
    let mut host = backend.inner.context.allocate_pinned(1)?;
    backend.inner.stream.copy_to_host(token, &mut host)?;
    Ok(host.to_vec()?[0])
}

fn table(block: BlockId, tokens: usize) -> BlockTable {
    let mut table = BlockTable::with_block_size(16);
    table.push(block);
    table.set_token_len(tokens);
    table
}

fn maximum(values: &[mircuda::bf16]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1.to_f32().total_cmp(&right.1.to_f32()).then_with(|| right.0.cmp(&left.0))
        })
        .map_or(0, |(index, _)| index)
}
