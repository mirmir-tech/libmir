use std::{fs, path::Path};

use models::{
    layout::{DecoderConfig, ModelLayout},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, WeightBindingPlan},
};
use runtime::kv::{BlockId, BlockTable, CacheConfig};
use uuid::Uuid;

use crate::{
    CudaBackend, CudaConfig, CudaModelSessionConfig, DenseSwiGluLayerLoadConfig, ProjectionFormat,
};

/// Decode steps checked after each prefill length.
const STEPS: usize = 4;
/// Logit margin within which a decode choice counts as a tie with prefill.
const TIE: f32 = 1.5;

/// Whole-model prefill of the reference prompt from
/// `benchmarks/tools/gemma/layer_oracle.py --reference`, compared by top
/// next-token ids with the Hugging Face layers.
#[test]
#[ignore = "requires GEMMA_DENSE_MODEL and GEMMA_MODEL_ORACLE"]
#[allow(clippy::print_stdout)]
fn dense_gemma_model_prefill_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::var("GEMMA_DENSE_MODEL")?;
    let oracle = std::env::var("GEMMA_MODEL_ORACLE")?;
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(Path::new(&oracle).join("meta.json"))?)?;
    let ids = meta["ids"]
        .as_array()
        .ok_or("meta lacks ids")?
        .iter()
        .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()).ok_or("bad id"))
        .collect::<Result<Vec<_>, _>>()?;
    let expected = meta["top_ids"][0].as_u64().ok_or("meta lacks top ids")?;
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let spec = SemanticModelSpec::from_layout(&layout, &decoder, &catalog)?;
    let bindings = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut ignored = |_completed, _detail| {};
    let template = backend.load_dense_swiglu_model_template_with_bindings(
        &decoder,
        &spec,
        &bindings,
        &catalog,
        DenseSwiGluLayerLoadConfig {
            cache: CacheConfig::new(4),
            max_sequence_blocks: 4,
            projection_format: ProjectionFormat::NvFp4,
        },
        &mut ignored,
    )?;
    let mut session = template.instantiate()?;
    let mut table = BlockTable::with_block_size(16);
    for block in 0..ids.len().div_ceil(16) {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(ids.len());
    let logits = session.prefill_from(Uuid::nil(), &ids, 0, &table)?;
    let mut host = backend.inner.context.allocate_pinned(logits.len())?;
    backend.inner.stream.copy_to_host(logits, &mut host)?;
    let logits = host.to_vec()?;
    let mut ranked = logits
        .iter()
        .enumerate()
        .map(|(id, value)| (id, value.to_f32()))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    println!("libmir top: {:?}", &ranked[..8]);
    println!("reference top: {}", meta["top_ids"]);
    assert_eq!(u64::try_from(ranked[0].0)?, expected);
    Ok(())
}

/// Decode after a prefill of `length` tokens must match a prefill of
/// `length + 1` tokens; the lengths straddle the sliding window.
#[test]
#[ignore = "requires GEMMA_DENSE_MODEL and GEMMA_LONG_REFERENCE"]
#[allow(clippy::print_stdout)]
fn dense_gemma_decode_matches_prefill_across_the_window() -> Result<(), Box<dyn std::error::Error>>
{
    let root = std::env::var("GEMMA_DENSE_MODEL")?;
    let reference: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(std::env::var("GEMMA_LONG_REFERENCE")?)?)?;
    let ids = reference["records"][0]["prompt_ids"]
        .as_array()
        .ok_or("reference lacks prompt ids")?
        .iter()
        .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()).ok_or("bad id"))
        .collect::<Result<Vec<_>, _>>()?;
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let spec = SemanticModelSpec::from_layout(&layout, &decoder, &catalog)?;
    let bindings = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
    // `GEMMA_UNTUNED` disables timing-based kernel selection, as `teacher-forcing
    // untuned`.
    let mut config = CudaConfig::default();
    if std::env::var_os("GEMMA_UNTUNED").is_some() {
        config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    }
    let backend = CudaBackend::new(config)?;
    let blocks = ids.len().div_ceil(16) + 1;
    let mut ignored = |_completed, _detail| {};
    let template = backend.load_dense_swiglu_model_template_with_bindings(
        &decoder,
        &spec,
        &bindings,
        &catalog,
        DenseSwiGluLayerLoadConfig {
            cache: CacheConfig::new(u32::try_from(blocks)?),
            max_sequence_blocks: blocks,
            projection_format: ProjectionFormat::NvFp4,
        },
        &mut ignored,
    )?;
    // The engine decodes through decode batches over the session's caches;
    // `GEMMA_SCALAR_DECODE` uses the session's own decode graph instead.
    let caches = template.allocate_shared_kv()?;
    let mut session =
        template.instantiate_with_config_and_caches(CudaModelSessionConfig::default(), &caches)?;
    session.prepare_packed_prefill_batch(1)?;
    // One bucket for every length, as the engine keeps it: its captured graph
    // must follow the split-attention partitions of the growing context.
    let mut batch = template.instantiate_decode_batch_with_caches(1, &caches)?;
    let scalar = std::env::var_os("GEMMA_SCALAR_DECODE").is_some();
    let mut failures = Vec::new();
    // The first decode captures a graph; later steps replay it. A fresh
    // session prefills each reference so its cache cannot leak into the next.
    for length in [500, 1000, 1100, 2000, ids.len() - STEPS] {
        // The engine prefills in packed chunks, each starting where the last ended.
        let chunk = CudaModelSessionConfig::default().prefill_chunk_tokens;
        for start in (0..length).step_by(chunk) {
            let end = (start + chunk).min(length);
            session.prefill_packed_chunk(
                &ids[start..end],
                &[&table(end)?],
                &[start],
                &[end - start],
            )?;
        }
        for step in 0..STEPS {
            let position = length + step;
            let next = table(position + 1)?;
            let stepped = if scalar {
                logits(&backend, session.decode(Uuid::nil(), ids[position], &next)?)?
            } else {
                logits(&backend, batch.decode(&[ids[position]], &[&next])?)?
            };
            let mut fresh = template.instantiate()?;
            let prefilled = logits(
                &backend,
                fresh.prefill_from(Uuid::nil(), &ids[..=position], 0, &table(position + 1)?)?,
            )?;
            let (choice, value) = top(&stepped).ok_or("empty decode logits")?;
            let (expected, best) = top(&prefilled).ok_or("empty prefill logits")?;
            println!(
                "length {length} step {step}: decode ({choice}, {value}) prefill ({expected}, {best})"
            );
            // The two paths quantize activations in different shapes, so a
            // near-tie may pick either token; anything else is a defect.
            if !stepped.iter().all(|value| value.is_finite()) || prefilled[choice] < best - TIE {
                failures.push((length, step));
            }
        }
    }
    assert!(failures.is_empty(), "decode leaves the prefill near-ties at {failures:?}");
    Ok(())
}

fn table(tokens: usize) -> Result<BlockTable, Box<dyn std::error::Error>> {
    let mut table = BlockTable::with_block_size(16);
    for block in 0..tokens.div_ceil(16) {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}

fn logits(
    backend: &CudaBackend,
    logits: &mircuda::DeviceBuffer<mircuda::bf16>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut host = backend.inner.context.allocate_pinned(logits.len())?;
    backend.inner.stream.copy_to_host(logits, &mut host)?;
    Ok(host.to_vec()?.iter().map(|value| value.to_f32()).collect())
}

fn top(logits: &[f32]) -> Option<(usize, f32)> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
}
