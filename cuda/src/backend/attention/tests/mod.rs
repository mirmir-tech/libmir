use std::path::Path;

use ::runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvStorageSpec, KvWritePlan};
use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};
use uuid::Uuid;

use super::*;
use crate::{Bf16LinearPackWeights, CudaConfig, CudaTensorSet};
mod batch;
mod io;
mod mrope;
pub(super) use io::read;
const BASE: &str = "model.language_model.layers.0";

#[test]
fn checkpoint_decode_attention_executes_direct_and_graphed_tokens()
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
    let mut attention = backend.prepare_decode_attention_bf16(config)?;
    let first = input(&backend, 0)?;
    let second = input(&backend, 1)?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 2_816)?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(1));

    table.set_token_len(1);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 1)?;
    attention.execute(&first, weights, &plan, &table, &mut output)?;
    let first_output = read(&backend, &output)?;
    table.set_token_len(2);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 1, 1)?;
    let mut graph = attention.capture(&second, weights, &plan, &table, &output)?;
    drop(tensors);
    graph.launch()?;
    let second_output = read(&backend, &output)?;
    table.set_token_len(3);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 2, 1)?;
    graph.execute(&plan, &table)?;
    let third_output = read(&backend, &output)?;
    assert_valid(&first_output);
    assert_valid(&second_output);
    assert_valid(&third_output);
    assert!(first_output.iter().zip(&second_output).any(|(left, right)| left != right));
    assert!(second_output.iter().zip(&third_output).any(|(left, right)| left != right));
    let mut remapped = table.clone();
    remapped.push(BlockId(0));
    assert!(matches!(
        graph.execute(&plan, &remapped),
        Err(Error::InvalidPagedKv("captured attention requires KV block-table recapture"))
    ));
    Ok(())
}

#[test]
fn checkpoint_prefill_attention_matches_token_steps()
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
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(1));

    let mut sequential = backend.prepare_decode_attention_bf16(config())?;
    let mut token_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 2_816)?;
    table.set_token_len(1);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 1)?;
    sequential.execute(&input(&backend, 0)?, weights, &plan, &table, &mut token_output)?;
    let first = read(&backend, &token_output)?;
    table.set_token_len(2);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 1, 1)?;
    sequential.execute(&input(&backend, 1)?, weights, &plan, &table, &mut token_output)?;
    let second = read(&backend, &token_output)?;

    let mut state = backend.prepare_decode_attention_bf16(config())?;
    let mut prefill = backend.prepare_prefill_attention_bf16(config(), 2)?;
    let mut batch_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 5_632)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 2)?;
    prefill.execute(
        &mut state,
        &input_batch(&backend, 2)?,
        weights,
        &plan,
        &table,
        0,
        &mut batch_output,
    )?;
    let actual = read(&backend, &batch_output)?;
    assert_close(&actual[..2_816], &first);
    assert_close(&actual[2_816..], &second);
    Ok(())
}

pub(super) fn config() -> DecodeAttentionConfig {
    DecodeAttentionConfig {
        layer: 0,
        hidden_size: 2_816,
        query_heads: 16,
        rotary_dim: 256,
        rope_pairing_dim: 256,
        rope_theta: 10_000.0,
        rms_norm_epsilon: 1.0e-6,
        attention_scale: 0.0625,
        projection_format: ProjectionFormat::Bf16,
        qkv_normalization: crate::kernels::QkvNormalization::ALL,
        sliding_window: Some(1_024),
        max_sequence_blocks: 2,
        cache: KvStorageSpec::new(
            CacheConfig {
                block_size: 16,
                block_count: 2,
                dtype: KvCacheDType::BFloat16,
            },
            8,
            256,
        ),
    }
}

pub(super) fn names() -> [String; 7] {
    [
        format!("{BASE}.input_layernorm.weight"),
        format!("{BASE}.self_attn.q_proj.weight"),
        format!("{BASE}.self_attn.k_proj.weight"),
        format!("{BASE}.self_attn.v_proj.weight"),
        format!("{BASE}.self_attn.q_norm.weight"),
        format!("{BASE}.self_attn.k_norm.weight"),
        format!("{BASE}.self_attn.o_proj.weight"),
    ]
}

pub(super) fn weights<'a>(
    tensors: &'a CudaTensorSet,
    names: &[String; 7],
    qkv: &'a Bf16LinearPackWeights<3>,
) -> Result<DecodeAttentionWeights<'a>> {
    let get = |index: usize| {
        tensors
            .get(names[index].as_str())
            .ok_or_else(|| Error::MissingTensor(names[index].clone()))
    };
    Ok(DecodeAttentionWeights {
        input_norm: get(0)?,
        qkv: DecodeQkvWeights::Bf16(qkv),
        query_norm: get(4)?
            .as_bf16()
            .ok_or_else(|| Error::DTypeMismatch { name: names[4].clone(), expected: "BF16" })?,
        key_norm: get(5)?
            .as_bf16()
            .ok_or_else(|| Error::DTypeMismatch { name: names[5].clone(), expected: "BF16" })?,
        output: DecodeAttentionOutputWeight::Bf16(get(6)?),
    })
}

pub(super) fn qkv(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    names: &[String; 7],
) -> Result<Bf16LinearPackWeights<3>> {
    let get = |index: usize| {
        tensors
            .get(names[index].as_str())
            .ok_or_else(|| Error::MissingTensor(names[index].clone()))
    };
    backend.pack_bf16_linears([get(1)?, get(2)?, get(3)?])
}

pub(super) fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}

pub(super) fn upload(backend: &CudaBackend, infos: &[&TensorInfo]) -> Result<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    for info in infos {
        upload.enqueue(info)?;
    }
    upload.finish()
}

pub(super) fn input(backend: &CudaBackend, offset: usize) -> Result<DeviceBuffer<bf16>> {
    let values = (0..2_816)
        .map(|index| {
            let value = f32::from(u8::try_from((index + offset) % 31)?) / 16.0 - 0.9375;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    copy(backend, &values)
}

pub(super) fn input_batch(backend: &CudaBackend, tokens: usize) -> Result<DeviceBuffer<bf16>> {
    let values = (0..tokens * 2_816)
        .map(|index| {
            let token = index / 2_816;
            let column = index % 2_816;
            let value = f32::from(u8::try_from((column + token) % 31)?) / 16.0 - 0.9375;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    copy(backend, &values)
}

pub(super) fn assert_close(actual: &[bf16], expected: &[bf16]) {
    assert_eq!(actual.len(), expected.len());
    let difference = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(difference < 0.5, "maximum BF16 attention difference: {difference}");
}

fn assert_valid(values: &[bf16]) {
    assert!(values.iter().all(|value| value.to_f32().is_finite()));
    assert!(values.iter().any(|value| value.to_f32() != 0.0));
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}
