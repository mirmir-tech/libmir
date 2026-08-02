#![cfg(target_os = "linux")]

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{
    layout::{DecoderConfig, ModelLayout},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo, WeightBindingPlan},
};
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType};

use super::*;
use crate::{
    CudaConfig, DensePlanRequest, DenseRole, DenseSwiGluLayerLoadConfig, ExecutionPhase,
    ProjectionFormat, kernels::QkvNormalization,
};

#[path = "tests/fp8_stages.rs"]
mod fp8_stages;
#[path = "tests/mxfp4.rs"]
mod mxfp4;

type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
#[ignore = "requires MIRMIR_FP8_MODEL"]
fn qwen2_stack_matches_dynamic_fp8_activation_reference() -> TestResult<()> {
    let root = std::env::var("MIRMIR_FP8_MODEL")?;
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let bindings = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let load = DenseSwiGluLayerLoadConfig {
        cache: CacheConfig {
            block_size: 16,
            block_count: 2,
            dtype: KvCacheDType::BFloat16,
        },
        max_sequence_blocks: 2,
        qkv_normalization: QkvNormalization::NONE,
        projection_format: ProjectionFormat::DirectFp8,
    };
    let input_values = embedding_row(&catalog, 785, decoder.hidden_size)?;
    let mut input = copy(&backend, &input_values)?;
    let mut output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, decoder.hidden_size)?;
    fp8_stages::validate(
        &backend,
        &catalog,
        bindings.dense_decoder_layer(0)?,
        &input,
        decoder.hidden_size,
    )?;
    for index in 0..decoder.num_hidden_layers {
        let (template, _bytes) = backend.load_dense_swiglu_layer_tracked(
            &decoder,
            &catalog,
            index,
            bindings.dense_decoder_layer(index)?,
            load,
        )?;
        let cache = backend.prepare_paged_kv(index, template.config().attention.cache)?;
        let mut layer = template.instantiate_with_cache(&input, &output, cache)?;
        let mut prefill = template.instantiate_prefill(1)?;
        let mut table = BlockTable::with_block_size(16);
        table.push(BlockId(0));
        table.set_token_len(1);
        let mut batch =
            backend.prepare_paged_prefill_batch(template.config().attention.cache, 2, 1, 1)?;
        batch.prepare(&[&table], &[0], &[1])?;
        prefill.execute_batch(&mut layer, &input, template.weights(), &batch, &mut output)?;
        if index == 0 {
            let layer_values = read(&backend, &output)?;
            assert_reference(&layer_values[..16], &layer_reference(), "layer 0");
        }
        std::mem::swap(&mut input, &mut output);
    }
    assert_reference(&read(&backend, &input)?[..16], &stack_reference(), "layer stack");
    validate_boundary(&backend, &catalog, &input, decoder.hidden_size, decoder.vocab_size)?;
    Ok(())
}

fn validate_boundary(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    hidden: &DeviceBuffer<bf16>,
    features: usize,
    vocab: usize,
) -> TestResult<()> {
    let mut upload = backend.begin_tensor_upload();
    upload.enqueue(required(catalog, "model.norm.weight")?)?;
    upload.enqueue(required(catalog, "model.embed_tokens.weight")?)?;
    let tensors = upload.finish()?;
    let norm_weight = tensors.get("model.norm.weight").ok_or("missing uploaded final norm")?;
    let output_weight =
        tensors.get("model.embed_tokens.weight").ok_or("missing uploaded output head")?;
    let selected = copy(backend, &[785_u32])?;
    let mut embedded = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, features)?;
    backend
        .prepare_bf16_embedding(vocab, features, 1.0)?
        .execute(&selected, 0, output_weight, &mut embedded)?;
    let expected_embedding = embedding_row(catalog, 785, features)?;
    assert_eq!(
        read(backend, &embedded)?,
        expected_embedding,
        "Qwen2 embedding gather differs from checkpoint row"
    );
    let mut normalized = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, features)?;
    backend
        .prepare_rms_norm_bf16(1, features, 1.0e-6)?
        .execute(hidden, norm_weight, &mut normalized)?;
    assert_reference(&read(backend, &normalized)?[..16], &norm_reference(), "final norm");
    let mut logits = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, vocab)?;
    backend
        .prepare_bf16_projection(DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role: DenseRole::OutputHead,
            tokens: 1,
            input_features: features,
            output_features: vocab,
        })?
        .execute(&normalized, output_weight, &mut logits)?;
    let logits = read(backend, &logits)?;
    let mut top = logits.iter().enumerate().collect::<Vec<_>>();
    top.sort_unstable_by(|left, right| right.1.to_f32().total_cmp(&left.1.to_f32()));
    let actual = top
        .iter()
        .take(8)
        .map(|(token, score)| (*token, score.to_f32()))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [
            (6364, 12.125),
            (431, 12.0625),
            (730, 11.0),
            (1205, 10.8125),
            (356, 10.5625),
            (7071, 10.4375),
            (393, 10.125),
            (434, 10.0625),
        ],
        "Qwen2 output-head ranking differs from vLLM"
    );
    Ok(())
}

fn assert_reference(actual: &[bf16], expected: &[f32], label: &str) {
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum <= f32::EPSILON, "Qwen2 {label} maximum reference error: {maximum}");
}

fn layer_reference() -> [f32; 16] {
    [
        -0.328_125, -0.188_476_56, -0.065_429_69, -0.176_757_81, 0.046_386_72, -0.013_061_523,
        0.056_640_625, 0.019_775_39, 0.024_780_273, 0.044_433_594, -0.039_062_5, 0.124_023_44,
        0.115_722_656, -0.001_220_703, -0.053_710_938, 0.066_894_53,
    ]
}

fn stack_reference() -> [f32; 16] {
    [
        -0.507_812_5, -0.075_195_31, 1.164_062_5, -1.265_625, -1.984_375, -0.441_406_25, 1.156_25,
        0.490_234_38, 1.406_25, 0.898_437_5, 0.048_828_125, 1.312_5, -0.703_125, -3.906_25,
        1.617_187_5, 1.125,
    ]
}

fn norm_reference() -> [f32; 16] {
    [
        -1.281_25, -0.171_875, 2.796_875, -2.937_5, -4.375, -0.988_281_25, 2.671_875, 1.093_75,
        3.25, 2.046_875, 0.113_769_53, 2.921_875, -1.648_437_5, -8.437_5, 3.671_875, 2.578_125,
    ]
}

fn embedding_row(catalog: &TensorCatalog, token: usize, hidden: usize) -> TestResult<Vec<bf16>> {
    let info = required(catalog, "model.embed_tokens.weight")?;
    let row_bytes = hidden.checked_mul(2).ok_or("embedding row size overflow")?;
    let offset = info
        .payload_start()?
        .checked_add(u64::try_from(token.checked_mul(row_bytes).ok_or("token row overflow")?)?)
        .ok_or("embedding row offset overflow")?;
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0; row_bytes];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)))
        .collect())
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> TestResult<&'a TensorInfo> {
    catalog.get(name).ok_or_else(|| format!("missing {name}").into())
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, values: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    backend.inner.stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
