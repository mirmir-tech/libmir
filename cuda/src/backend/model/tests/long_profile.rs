use std::path::Path;

use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable},
};
use uuid::Uuid;

use crate::{
    CudaAttentionPolicy, CudaBackend, CudaConfig, CudaModelSessionConfig, CudaPlanningPolicy,
    ProjectionFormat, Result,
};

const DEFAULT_CONTEXT_TOKENS: usize = 512;
const DECODE_TOKENS: usize = 64;
const BLOCK_SIZE: usize = 16;
const BATCH_SIZE: usize = 10;

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn profile_long_context_model_decode() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_LONG_DECODE").is_none() {
        return Ok(());
    }
    let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL")
        .or_else(|| std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL"))
    else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(profile_config()?)?;
    let context_tokens = std::env::var("LIBMIR_CUDA_PROFILE_CONTEXT")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT_TOKENS);
    let block_count = (context_tokens + DECODE_TOKENS).div_ceil(BLOCK_SIZE);
    let template = super::projection_gate::load_template(&backend, &decoder, &catalog)?;
    let mut session =
        template.instantiate_with_config(CudaModelSessionConfig { prefill_chunk_tokens: 128 })?;
    let mut table = block_table(block_count)?;
    let prompt = (0..context_tokens)
        .map(|index| u32::try_from(index % 1_024 + 2))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    table.set_token_len(context_tokens);
    session.prefill_from(Uuid::nil(), &prompt, 0, &table)?;
    for offset in 1..=2 {
        table.set_token_len(context_tokens + offset);
        session.decode(Uuid::nil(), 2, &table)?;
    }
    backend.inner.stream.synchronize()?;
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    let profiler = std::env::var_os("LIBMIR_CUDA_PROFILE_DECODE")
        .is_some()
        .then(|| backend.inner.context.start_profiler_range())
        .transpose()?;
    started.record(&backend.inner.stream)?;
    for offset in 3..=DECODE_TOKENS {
        table.set_token_len(context_tokens + offset);
        session.decode(Uuid::nil(), 2, &table)?;
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    if let Some(profiler) = profiler {
        profiler.stop()?;
    }
    let measured = (DECODE_TOKENS - 2) as f32;
    let elapsed = started.elapsed_ms(&completed)?;
    eprintln!(
        "full model decode at {context_tokens} tokens: {:.3} ms/token, {:.2} tok/s",
        elapsed / measured,
        measured * 1_000.0 / elapsed,
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn profile_long_context_model_batch_decode() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    if std::env::var_os("LIBMIR_CUDA_PROFILE_BATCH_DECODE").is_none() {
        return Ok(());
    }
    let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(profile_config()?)?;
    let context_tokens = std::env::var("LIBMIR_CUDA_PROFILE_CONTEXT")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT_TOKENS);
    let sequence_blocks = (context_tokens + DECODE_TOKENS).div_ceil(BLOCK_SIZE);
    let template = super::projection_gate::load_template_with_cache(
        &backend,
        &decoder,
        &catalog,
        ProjectionFormat::Bf16,
        sequence_blocks * BATCH_SIZE,
        sequence_blocks,
    )?;
    let caches = template.allocate_shared_kv()?;
    let mut scalar =
        template.instantiate_with_config_and_caches(CudaModelSessionConfig::default(), &caches)?;
    let prompt = (0..context_tokens)
        .map(|index| u32::try_from(index % 1_024 + 2))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let sessions = (0..BATCH_SIZE).map(|_| Uuid::new_v4()).collect::<Vec<_>>();
    let mut tables = (0..BATCH_SIZE)
        .map(|row| batch_table(row * sequence_blocks, sequence_blocks, context_tokens))
        .collect::<Result<Vec<_>>>()?;
    for (session, table) in sessions.iter().copied().zip(&tables) {
        scalar.prefill_from(session, &prompt, 0, table)?;
    }
    let mut batch = template.instantiate_decode_batch_with_caches(BATCH_SIZE, &caches)?;
    let tokens = vec![2_u32; BATCH_SIZE];
    let sampling = (0..BATCH_SIZE)
        .map(|row| SamplingLogits::Sample {
            vocab_size: decoder.vocab_size,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            draw: (row + 1) as f32 / (BATCH_SIZE + 1) as f32,
        })
        .collect::<Vec<_>>();
    let profile_sampling = std::env::var_os("LIBMIR_CUDA_PROFILE_BATCH_SAMPLE").is_some();
    let profile_readback = std::env::var_os("LIBMIR_CUDA_PROFILE_BATCH_READBACK").is_some();
    for offset in 1..=2 {
        for table in &mut tables {
            table.set_token_len(context_tokens + offset);
        }
        let references = tables.iter().collect::<Vec<_>>();
        batch.decode(&tokens, &references)?;
        if profile_sampling {
            let selected = batch.sample(&sampling)?;
            if profile_readback {
                backend.read_tokens(selected)?;
            }
        }
    }
    backend.inner.stream.synchronize()?;
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    let profiler = backend.inner.context.start_profiler_range()?;
    started.record(&backend.inner.stream)?;
    for offset in 3..=DECODE_TOKENS {
        for table in &mut tables {
            table.set_token_len(context_tokens + offset);
        }
        let references = tables.iter().collect::<Vec<_>>();
        batch.decode(&tokens, &references)?;
        if profile_sampling {
            let selected = batch.sample(&sampling)?;
            if profile_readback {
                backend.read_tokens(selected)?;
            }
        }
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    profiler.stop()?;
    let measured = (DECODE_TOKENS - 2) * BATCH_SIZE;
    let elapsed = started.elapsed_ms(&completed)?;
    eprintln!(
        "full model batch-{BATCH_SIZE} decode{} at {context_tokens} tokens: \
         {:.3} ms/step, {:.2} aggregate tok/s",
        match (profile_sampling, profile_readback) {
            (true, true) => " with sampling and readback",
            (true, false) => " with sampling",
            (false, _) => "",
        },
        elapsed / (DECODE_TOKENS - 2) as f32,
        measured as f32 * 1_000.0 / elapsed,
    );
    Ok(())
}

fn profile_config() -> std::result::Result<CudaConfig, Box<dyn std::error::Error>> {
    let attention = match std::env::var("LIBMIR_CUDA_PROFILE_ATTENTION").ok().as_deref() {
        None | Some("auto") => CudaAttentionPolicy::Auto,
        Some("direct") => CudaAttentionPolicy::Direct,
        Some(value) => {
            let partition_tokens = value.parse::<usize>()?;
            CudaAttentionPolicy::SplitKv {
                partition_tokens,
                threshold_tokens: partition_tokens.saturating_add(1),
            }
        },
    };
    Ok(CudaConfig {
        planning: CudaPlanningPolicy {
            attention,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })
}

fn block_table(blocks: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    Ok(table)
}

fn batch_table(first: usize, blocks: usize, tokens: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in first..first + blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}
