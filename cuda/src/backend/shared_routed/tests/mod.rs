mod fixture;

use mircuda::{DeviceElement, bf16};
use models::layout::DecoderConfig;
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType};
use serde_json::json;
use uuid::Uuid;

use super::*;
use crate::{CudaConfig, Result};

#[test]
fn executes_mixed_linear_and_full_attention_session() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let decoder = decoder()?;
    let fixture = fixture::HybridFixture::new(&decoder)?;
    let cache = CacheConfig {
        block_size: 16,
        block_count: 4,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig { cache, max_sequence_blocks: 2 },
    )?;
    let mut session = template.instantiate()?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(0));

    table.set_token_len(2);
    let logits = session.prefill(Uuid::nil(), &[1, 2], &table)?;
    assert!(read(&backend, logits)?.iter().all(|value| *value == bf16::ZERO));
    assert_eq!(session.position(), 2);

    table.set_token_len(3);
    let logits = session.decode(Uuid::nil(), 3, &table)?;
    assert!(read(&backend, logits)?.iter().all(|value| *value == bf16::ZERO));
    assert_eq!(session.position(), 3);
    assert_eq!(read(&backend, session.sample(runtime::backend::SamplingLogits::None)?)?, [0]);
    Ok(())
}

#[test]
fn continues_decode_after_spatial_vision_prefill() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let decoder = decoder()?;
    let fixture = fixture::HybridFixture::new(&decoder)?;
    let cache = CacheConfig {
        block_size: 16,
        block_count: 4,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig { cache, max_sequence_blocks: 2 },
    )?;
    let mut session = template.instantiate()?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(0));
    table.set_token_len(3);
    let image = copy(&backend, &vec![bf16::ZERO; decoder.hidden_size])?;
    let positions = [0_u32, 1, 2, 0, 1, 2, 0, 1, 2];
    session.prefill_vision(Uuid::nil(), &[1, 2, 3], &positions, &table, (1, 2), &image, -1)?;
    assert_eq!(session.position(), 3);
    assert_eq!(session.position_delta(), -1);

    table.set_token_len(4);
    let logits = session.decode(Uuid::nil(), 4, &table)?;
    assert!(read(&backend, logits)?.iter().all(|value| *value == bf16::ZERO));
    assert_eq!(session.position(), 4);
    Ok(())
}

fn decoder() -> Result<DecoderConfig> {
    Ok(DecoderConfig::from_value(&json!({
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 128,
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "hidden_act": "silu",
            "attn_output_gate": true,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 64,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 1],
                "full_attention": {"partial_rotary_factor": 0.1875}
            }
        }
    }))?)
}

fn read<T: DeviceElement>(
    backend: &CudaBackend,
    source: &mircuda::DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<mircuda::DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
