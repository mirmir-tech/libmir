mod fixture;

use ::runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvStorageSpec, KvWritePlan};
use mircuda::{DeviceBuffer, DeviceElement, bf16};
use uuid::Uuid;

use super::super::super::*;
use crate::{CudaConfig, Result, kernels::GatedAttentionSplit};

#[test]
fn splits_query_and_gate_per_attention_head() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let input = copy(&backend, &bf16s(&[1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0]))?;
    let mut query = allocate(&backend, 4)?;
    let mut gate = allocate(&backend, 4)?;
    GatedAttentionSplit::compile(&backend.inner.compiler, 1, 2, 2)?.execute(
        &backend.inner.stream,
        &input,
        &mut query,
        &mut gate,
    )?;
    assert_eq!(floats(&read(&backend, &query)?), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(floats(&read(&backend, &gate)?), vec![11.0, 12.0, 13.0, 14.0]);
    Ok(())
}

#[test]
fn executes_affine_gated_attention_prefill_and_decode() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = config();
    let fixture = fixture::AttentionFixture::new(config)?;
    let tensors = fixture.upload(&backend)?;
    let layer =
        CudaAffineGatedFullAttention::from_tensors(&backend, &tensors, fixture::PREFIX, config)?;
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: 16,
            block_count: 2,
            dtype: KvCacheDType::BFloat16,
        },
        config.key_value_heads,
        config.head_dim,
    );
    let mut state = layer.prepare_state(0, storage, 2)?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(0));

    table.set_token_len(2);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 2)?;
    let input = copy(&backend, &vec![bf16::ZERO; 2 * config.hidden_size])?;
    let positions = copy(&backend, &[0_u32, 1, 0, 1, 0, 1])?;
    let mut output = allocate(&backend, 2 * config.hidden_size)?;
    layer
        .prepare(2)?
        .execute(&input, &positions, &mut state, &plan, &table, 0, None, &mut output)?;
    assert!(read(&backend, &output)?.iter().all(|value| *value == bf16::ZERO));

    table.set_token_len(3);
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 2, 1)?;
    let input = copy(&backend, &vec![bf16::ZERO; config.hidden_size])?;
    let positions = copy(&backend, &[2_u32, 2, 2])?;
    let mut output = allocate(&backend, config.hidden_size)?;
    layer
        .prepare(1)?
        .execute(&input, &positions, &mut state, &plan, &table, 2, None, &mut output)?;
    assert!(read(&backend, &output)?.iter().all(|value| *value == bf16::ZERO));
    Ok(())
}

fn config() -> AffineGatedFullAttentionConfig {
    AffineGatedFullAttentionConfig {
        hidden_size: 64,
        query_heads: 2,
        key_value_heads: 1,
        head_dim: 32,
        rotary_dim: 6,
        rope_sections: [1, 1, 1],
        rope_interleaved: true,
        rope_theta: 10_000.0,
        attention_scale: 32.0_f32.sqrt().recip(),
        rms_norm_epsilon: 1.0e-6,
        norm_weight_shift: 0.0,
        group_size: 64,
        bits: 4,
    }
}

fn allocate(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    Ok(backend.inner.pool.allocate(&backend.inner.stream, elements)?)
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn floats(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
