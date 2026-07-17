use std::path::Path;

use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::kv::{BlockId, BlockTable};
use uuid::Uuid;

use crate::{
    CudaAttentionPolicy, CudaBackend, CudaConfig, CudaModelSessionConfig, CudaPlanningPolicy,
    Result,
};

const DEFAULT_CONTEXT_TOKENS: usize = 512;
const DECODE_TOKENS: usize = 64;
const BLOCK_SIZE: usize = 16;

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
