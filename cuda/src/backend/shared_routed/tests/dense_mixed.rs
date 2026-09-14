use super::*;

#[test]
fn dense_mixed_prefill_decode_and_checkpoint_continuation() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut decoder = decoder()?;
    decoder.num_experts = None;
    decoder.top_k_experts = None;
    decoder.moe_intermediate_size = None;
    decoder.shared_expert_intermediate_size = None;
    decoder.intermediate_size = 64;
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
    let checkpoint = session.checkpoint()?;
    table.set_token_len(3);
    let logits = session.decode(Uuid::nil(), 3, &table)?;
    assert!(read(&backend, logits)?.iter().all(|value| *value == bf16::ZERO));
    let first = read(&backend, logits)?;
    session.restore_checkpoint(&checkpoint)?;
    assert_eq!(session.position(), 2);
    let logits = session.decode(Uuid::nil(), 3, &table)?;
    assert_eq!(read(&backend, logits)?, first);
    assert_eq!(session.position(), 3);
    Ok(())
}
