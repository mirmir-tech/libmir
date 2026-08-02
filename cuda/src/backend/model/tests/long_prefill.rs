use std::path::Path;

use models::{
    layout::{DecoderConfig, ModelLayout},
    semantic::SemanticModelSpec,
    weights::TensorCatalog,
};
use runtime::kv::{BlockId, BlockTable, CacheConfig};
use uuid::Uuid;

use crate::{
    CudaBackend, CudaConfig, CudaModelSessionConfig, CudaMoeModelSession,
    DenseSwiGluLayerLoadConfig, ProjectionFormat, Result,
};

const DEFAULT_CONTEXT_TOKENS: usize = 4_096;
const DEFAULT_BATCH_SIZE: usize = 10;
const DEFAULT_BATCH_CHUNK: usize = 512;
const BLOCK_SIZE: usize = 16;

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn profile_long_context_model_prefill() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_LONG_PREFILL").is_none() {
        return Ok(());
    }
    let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let context_tokens = std::env::var("LIBMIR_CUDA_PROFILE_CONTEXT")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT_TOKENS);
    let chunk_tokens = std::env::var("LIBMIR_CUDA_PROFILE_PREFILL_CHUNK")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or_else(|| CudaModelSessionConfig::default().prefill_chunk_tokens);
    let block_count = context_tokens.div_ceil(BLOCK_SIZE);
    let backend = CudaBackend::new(CudaConfig::default())?;
    let template = load_template(&backend, &decoder, &catalog, block_count, block_count)?;
    let mut session = template
        .instantiate_with_config(CudaModelSessionConfig { prefill_chunk_tokens: chunk_tokens })?;
    let table = block_table(context_tokens, block_count)?;
    let prompt = (0..context_tokens)
        .map(|index| u32::try_from(index % 1_024 + 2))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    session.prefill_from(Uuid::nil(), &prompt, 0, &table)?;
    backend.inner.stream.synchronize()?;

    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    let profiler = backend.inner.context.start_profiler_range()?;
    started.record(&backend.inner.stream)?;
    session.prefill_from(Uuid::nil(), &prompt, 0, &table)?;
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    profiler.stop()?;
    let elapsed = started.elapsed_ms(&completed)?;
    eprintln!(
        "full model prefill at {context_tokens} tokens in chunks of {chunk_tokens}: \
         {elapsed:.3} ms, {:.2} tok/s",
        context_tokens as f32 * 1_000.0 / elapsed,
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn profile_long_context_model_batch_prefill() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    if std::env::var_os("LIBMIR_CUDA_PROFILE_BATCH_PREFILL").is_none() {
        return Ok(());
    }
    let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let context_tokens = environment_usize("LIBMIR_CUDA_PROFILE_CONTEXT", 2_048)?;
    let batch_size = environment_usize("LIBMIR_CUDA_PROFILE_BATCH", DEFAULT_BATCH_SIZE)?;
    let chunk_tokens = environment_usize("LIBMIR_CUDA_PROFILE_PREFILL_CHUNK", DEFAULT_BATCH_CHUNK)?;
    let profile_full = std::env::var_os("LIBMIR_CUDA_PROFILE_BATCH_FULL").is_some();
    let sequence_blocks = context_tokens.div_ceil(BLOCK_SIZE);
    let cache_blocks = sequence_blocks
        .checked_mul(batch_size)
        .ok_or("batch prefill cache size overflow")?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let template = load_template(&backend, &decoder, &catalog, cache_blocks, sequence_blocks)?;
    let caches = template.allocate_shared_kv()?;
    let capacity = chunk_tokens.checked_mul(batch_size).ok_or("batch prefill size overflow")?;
    let mut session = template.instantiate_with_config_and_caches(
        CudaModelSessionConfig { prefill_chunk_tokens: capacity },
        &caches,
    )?;
    let mut tables = (0..batch_size)
        .map(|row| batch_table(sequence_blocks, row * sequence_blocks))
        .collect::<Result<Vec<_>>>()?;
    let final_start = context_tokens.saturating_sub(1) / chunk_tokens * chunk_tokens;
    if profile_full {
        packed_sequence(&mut session, &mut tables, context_tokens, chunk_tokens)?;
    } else {
        for start in (0..final_start).step_by(chunk_tokens) {
            packed_round(&mut session, &mut tables, start, chunk_tokens)?;
        }
    }
    backend.inner.stream.synchronize()?;
    let final_tokens = context_tokens - final_start;
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    let profiler = backend.inner.context.start_profiler_range()?;
    started.record(&backend.inner.stream)?;
    if profile_full {
        packed_sequence(&mut session, &mut tables, context_tokens, chunk_tokens)?;
    } else {
        packed_round(&mut session, &mut tables, final_start, final_tokens)?;
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    profiler.stop()?;
    let elapsed = started.elapsed_ms(&completed)?;
    let profiled_tokens = if profile_full {
        context_tokens
    } else {
        final_tokens
    };
    eprintln!(
        "full model batch-{batch_size} prefill at {context_tokens} tokens \
         with {profiled_tokens} profiled tokens/row: {elapsed:.3} ms, {:.2} query tok/s",
        (batch_size * profiled_tokens) as f32 * 1_000.0 / elapsed,
    );
    Ok(())
}

fn load_template(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
    cache_blocks: usize,
    sequence_blocks: usize,
) -> Result<crate::CudaMoeModelTemplate> {
    let semantic = SemanticModelSpec::discover(decoder, catalog)?;
    let plan = crate::engine::lowering::CudaDecoderPlan::lower(&semantic);
    let mut ignored = |_completed, _detail| {};
    backend.load_dense_swiglu_model_template_with_progress(
        decoder,
        catalog,
        DenseSwiGluLayerLoadConfig {
            cache: CacheConfig::new(u32::try_from(cache_blocks)?),
            max_sequence_blocks: sequence_blocks,
            qkv_normalization: crate::engine::lowering::graph_normalization(&plan)?,
            projection_format: ProjectionFormat::Bf16,
        },
        &mut ignored,
    )
}

fn packed_round(
    session: &mut CudaMoeModelSession,
    tables: &mut [BlockTable],
    start: usize,
    tokens_per_row: usize,
) -> Result<()> {
    for table in tables.iter_mut() {
        table.set_token_len(start + tokens_per_row);
    }
    let tokens = (0..tables.len() * tokens_per_row)
        .map(|index| u32::try_from(index % 1_024 + 2))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let references = tables.iter().collect::<Vec<_>>();
    let starts = vec![start; tables.len()];
    let query_tokens = vec![tokens_per_row; tables.len()];
    session.prefill_packed_chunk(&tokens, &references, &starts, &query_tokens)
}

fn packed_sequence(
    session: &mut CudaMoeModelSession,
    tables: &mut [BlockTable],
    context_tokens: usize,
    chunk_tokens: usize,
) -> Result<()> {
    for start in (0..context_tokens).step_by(chunk_tokens) {
        packed_round(session, tables, start, chunk_tokens.min(context_tokens - start))?;
    }
    Ok(())
}

fn environment_usize(
    name: &'static str,
    default: usize,
) -> std::result::Result<usize, Box<dyn std::error::Error>> {
    let value = std::env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default);
    if value == 0 {
        Err(format!("{name} must be positive").into())
    } else {
        Ok(value)
    }
}

fn block_table(tokens: usize, blocks: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}

fn batch_table(blocks: usize, first: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in first..first + blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    Ok(table)
}
