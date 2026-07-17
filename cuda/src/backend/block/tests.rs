use std::path::Path;

use ::runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvWritePlan};
use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use uuid::Uuid;

use super::*;
use crate::{CudaConfig, NvFp4MoeLayerLoadConfig};

pub(super) const LAYER: usize = 5;

#[test]
fn checkpoint_full_decode_block_executes_direct_and_graphed_tokens()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let template =
        backend.load_nvfp4_moe_layer_template(&decoder, &catalog, LAYER, load_config())?;
    let output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 2_816)?;
    let input = input(&backend, 0)?;
    let mut executor = template.instantiate(&input, &output)?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(1));

    table.set_token_len(1);
    let plan = KvWritePlan::prefill(Uuid::nil(), LAYER, &table, 0, 1)?;
    assert_eq!(executor.execute(&plan, &table)?, DecodeGraphAction::CapturedAfterDirect);
    let first_output = read(&backend, &output)?;
    table.set_token_len(2);
    let plan = KvWritePlan::prefill(Uuid::nil(), LAYER, &table, 1, 1)?;
    overwrite_input(&backend, &input, 1)?;
    assert_eq!(executor.execute(&plan, &table)?, DecodeGraphAction::Replayed);
    let second_output = read(&backend, &output)?;
    table.set_token_len(3);
    let plan = KvWritePlan::prefill(Uuid::nil(), LAYER, &table, 2, 1)?;
    overwrite_input(&backend, &input, 2)?;
    assert_eq!(executor.execute(&plan, &table)?, DecodeGraphAction::Replayed);
    let third_output = read(&backend, &output)?;
    valid(&first_output);
    valid(&second_output);
    valid(&third_output);
    assert!(first_output.iter().zip(&second_output).any(|(left, right)| left != right));
    assert!(second_output.iter().zip(&third_output).any(|(left, right)| left != right));
    let mut remapped = table.clone();
    remapped.push(BlockId(0));
    remapped.set_token_len(17);
    let remapped_plan = KvWritePlan::prefill(Uuid::nil(), LAYER, &remapped, 16, 1)?;
    assert_eq!(executor.execute(&remapped_plan, &remapped)?, DecodeGraphAction::Recaptured);
    valid(&read(&backend, &output)?);
    Ok(())
}

#[test]
fn checkpoint_full_prefill_block_matches_token_steps()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let template =
        backend.load_nvfp4_moe_layer_template(&decoder, &catalog, LAYER, load_config())?;
    let token_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 2_816)?;
    let token_input = input(&backend, 0)?;
    let mut sequential = template.instantiate(&token_input, &token_output)?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(1));
    table.set_token_len(1);
    sequential.execute(&KvWritePlan::prefill(Uuid::nil(), LAYER, &table, 0, 1)?, &table)?;
    let first = read(&backend, &token_output)?;
    overwrite_input(&backend, &token_input, 1)?;
    table.set_token_len(2);
    sequential.execute(&KvWritePlan::prefill(Uuid::nil(), LAYER, &table, 1, 1)?, &table)?;
    let second = read(&backend, &token_output)?;

    let state_input = input(&backend, 0)?;
    let state_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 2_816)?;
    let mut state = template.instantiate(&state_input, &state_output)?;
    let mut prefill = template.instantiate_prefill(2)?;
    let batch_input = copy(&backend, &[input_values(0)?, input_values(1)?].concat())?;
    let mut batch_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 5_632)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), LAYER, &table, 0, 2)?;
    state.execute_prefill(&mut prefill, &batch_input, &plan, &table, 0, &mut batch_output)?;
    let actual = read(&backend, &batch_output)?;
    close(&actual[..2_816], &first);
    close(&actual[2_816..], &second);
    Ok(())
}

pub(super) fn load_config() -> NvFp4MoeLayerLoadConfig {
    NvFp4MoeLayerLoadConfig {
        cache: CacheConfig {
            block_size: 16,
            block_count: 2,
            dtype: KvCacheDType::BFloat16,
        },
        max_sequence_blocks: 2,
    }
}

pub(super) fn input(backend: &CudaBackend, offset: usize) -> Result<DeviceBuffer<bf16>> {
    copy(backend, &input_values(offset)?)
}

fn overwrite_input(
    backend: &CudaBackend,
    target: &DeviceBuffer<bf16>,
    offset: usize,
) -> Result<()> {
    let values = input_values(offset)?;
    let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(&values)?;
    let mut target = target.clone();
    backend.inner.stream.copy_to_device(&mut host, &mut target)?;
    backend.synchronize()
}

pub(super) fn input_values(offset: usize) -> Result<Vec<bf16>> {
    (0..2_816)
        .map(|index| {
            Ok(bf16::from_f32(f32::from(u8::try_from((index + offset) % 31)?) / 16.0 - 0.9375))
        })
        .collect()
}

pub(super) fn copy<T: DeviceElement>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}

pub(super) fn read<T: DeviceElement>(
    backend: &CudaBackend,
    source: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}

fn valid(values: &[bf16]) {
    assert!(values.iter().all(|value| value.to_f32().is_finite()));
    assert!(values.iter().any(|value| value.to_f32() != 0.0));
}

pub(super) fn close(actual: &[bf16], expected: &[bf16]) {
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum < 8.0, "maximum BF16 block difference: {maximum}");
}
