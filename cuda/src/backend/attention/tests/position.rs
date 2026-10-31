//! V2 gate: one real Nemotron-H attention layer without positional encoding
//! against the FP32 reference written by
//! `benchmarks/tools/nemotron/attention_oracle.py`. The first tokens run as
//! one prefill chunk, the next as a one-row packed prefill and the rest as
//! single decode steps over the same paged K/V state.

use std::{
    fs,
    path::{Path, PathBuf},
};

use ::runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvWritePlan};
use mircuda::bf16;
use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::{StateSpaceHybridMixerBindings, TensorCatalog},
};
use uuid::Uuid;

use super::{copy, read};
use crate::{
    AttentionLayerPolicy, CudaBackend, CudaConfig, DecodeAttentionBf16, DecodeAttentionConfig,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights,
    DirectFp8CheckpointWeight, Error, PrefillAttentionBf16, ProjectionFormat, Result,
};

const BLOCK_SIZE: usize = 16;
const DECODED: usize = 4;

fn bf16s(directory: &Path, name: &str) -> Result<Vec<bf16>> {
    let bytes = fs::read(directory.join(name))?;
    Ok(bytes.as_chunks::<2>().0.iter().map(|pair| bf16::from_le_bytes(*pair)).collect())
}

#[test]
#[ignore = "requires NEMOTRON_MODEL and NEMOTRON_ATTENTION_ORACLE"]
fn unrotated_attention_matches_the_reference_on_a_real_layer() -> Result<()> {
    let (Some(model), Some(oracle)) = (
        std::env::var_os("NEMOTRON_MODEL").map(PathBuf::from),
        std::env::var_os("NEMOTRON_ATTENTION_ORACLE").map(PathBuf::from),
    ) else {
        return Ok(());
    };
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(oracle.join("meta.json"))?)
        .map_err(|_| Error::InvalidExecutionPlan("oracle meta.json is not valid JSON"))?;
    let block = meta["block"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(Error::InvalidExecutionPlan("oracle meta has no block"))?;
    let layout = ModelLayout::inspect(&model)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let semantic = &contract.semantic.decoder;
    let (logical, _) = semantic
        .block_layout
        .locate(&semantic.layers, block)
        .ok_or(Error::InvalidExecutionPlan("oracle block is not a paired layer block"))?;
    let bindings = contract.bindings.state_space_hybrid_layer(logical)?;
    let StateSpaceHybridMixerBindings::Attention(projections) = bindings.mixer else {
        return Err(Error::InvalidExecutionPlan("oracle block is not an attention mixer"));
    };
    let input = bf16s(&oracle, "input.bf16")?;
    let expected = bf16s(&oracle, "output.bf16")?;
    let hidden = semantic.hidden_size;
    let tokens = input.len() / hidden;
    let blocks = tokens.div_ceil(BLOCK_SIZE);
    let policy = AttentionLayerPolicy {
        hidden_size: hidden,
        projection_format: ProjectionFormat::DirectFp8,
        cache: CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: u32::try_from(blocks)?,
            dtype: KvCacheDType::BFloat16,
        },
        max_sequence_blocks: blocks,
    };
    let config = DecodeAttentionConfig::from_semantic(0, &semantic.layers[logical], policy)?;
    assert_eq!(config.rotary, None);

    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    let all = [
        bindings.input_norm,
        projections.query,
        projections.key,
        projections.value,
        projections.output,
    ];
    for name in all.iter().flat_map(|binding| binding.physical_sources()) {
        upload.enqueue(catalog.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?)?;
    }
    let tensors = upload.finish()?;
    let load = |binding| DirectFp8CheckpointWeight::load_binding(&tensors, binding);
    let [query, key, value, output_weight] =
        [projections.query, projections.key, projections.value, projections.output];
    let (query, key, value) = (load(query)?, load(key)?, load(value)?);
    let output_weight = load(output_weight)?;
    let unused_norm = copy(&backend, &vec![bf16::ZERO; config.cache.key_head_dim])?;
    let weights = DecodeAttentionWeights {
        input_norm: tensors
            .get(&bindings.input_norm.source)
            .ok_or_else(|| Error::MissingTensor(bindings.input_norm.source.clone()))?,
        qkv: DecodeQkvWeights::DirectFp8([&query, &key, &value]),
        query_norm: &unused_norm,
        key_norm: &unused_norm,
        output: DecodeAttentionOutputWeight::DirectFp8(&output_weight),
    };

    let prefilled = tokens - DECODED;
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for index in 0..blocks {
        table.push(BlockId(u32::try_from(index)?));
    }
    table.set_token_len(prefilled);
    let cache = backend.prepare_paged_kv(config.layer, config.cache)?;
    let mut state =
        DecodeAttentionBf16::new_with_cache_and_weights(&backend, config, cache, Some(weights))?;
    let mut prefill = PrefillAttentionBf16::new(&backend, config, prefilled, Some(weights))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, prefilled * hidden)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, prefilled)?;
    let chunk = copy(&backend, &input[..prefilled * hidden])?;
    prefill.execute(&mut state, &chunk, weights, &plan, &table, 0, &mut output)?;
    let mut actual = read(&backend, &output)?;
    // A repeated prefill rewrites the same K/V and must be bit-identical.
    prefill.execute(&mut state, &chunk, weights, &plan, &table, 0, &mut output)?;
    let repeat = read(&backend, &output)?;
    assert!(actual.iter().zip(&repeat).all(|(a, b)| a.to_bits() == b.to_bits()));
    let mut row = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
    // A packed pass with one row (one sequence, one uncached prompt token)
    // takes the paged prefill kernels, not decode.
    table.set_token_len(prefilled + 1);
    let mut single = PrefillAttentionBf16::new(&backend, config, 1, Some(weights))?;
    let mut batch = backend.prepare_paged_prefill_batch(config.cache, blocks, 1, 1)?;
    batch.prepare(&[&table], &[prefilled], &[1])?;
    let step = copy(&backend, &input[prefilled * hidden..(prefilled + 1) * hidden])?;
    single.execute_batch(&mut state, &step, weights, &batch, &mut row)?;
    actual.extend(read(&backend, &row)?);
    for token in prefilled + 1..tokens {
        table.set_token_len(token + 1);
        let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, token, 1)?;
        let step = copy(&backend, &input[token * hidden..(token + 1) * hidden])?;
        state.execute(&step, weights, &plan, &table, &mut row)?;
        actual.extend(read(&backend, &row)?);
    }
    assert_parity(&actual, &expected, hidden);
    Ok(())
}

/// Relative L2 error below 1% and cosine similarity of every token of at
/// least 0.999.
#[allow(clippy::print_stderr)]
fn assert_parity(actual: &[bf16], expected: &[bf16], row: usize) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut norm, mut worst) = (0.0_f64, 0.0_f64, 1.0_f64);
    for (actual, expected) in actual.chunks(row).zip(expected.chunks(row)) {
        let (mut dot, mut left, mut right) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (actual, expected) in actual.iter().zip(expected) {
            let (actual, expected) = (f64::from(actual.to_f32()), f64::from(expected.to_f32()));
            error = (actual - expected).mul_add(actual - expected, error);
            norm = expected.mul_add(expected, norm);
            dot = actual.mul_add(expected, dot);
            left = actual.mul_add(actual, left);
            right = expected.mul_add(expected, right);
        }
        worst = worst.min(dot / (left.sqrt() * right.sqrt()));
    }
    let relative = (error / norm).sqrt();
    eprintln!("relative L2 {relative:.5}, worst token cosine {worst:.6}");
    assert!(relative < 0.01, "relative L2 error {relative}");
    assert!(worst >= 0.999, "worst token cosine {worst}");
}
