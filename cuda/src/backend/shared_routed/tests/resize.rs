use super::*;

/// A template resized to more blocks serves the same prompt with the same
/// logits and reports the larger page allocation.
#[test]
fn resized_cache_serves_the_same_logits() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut decoder = decoder()?;
    decoder.head_dim = 64;
    decoder.full_attention_partial_rotary_factor = Some(0.09375);
    let fixture = fixture::HybridFixture::nonzero_routed(&decoder)?;
    let cache = CacheConfig {
        block_size: 16,
        block_count: 2,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig { cache, max_sequence_blocks: 2 },
    )?;
    let tokens = (0..24)
        .map(|index| u32::try_from(index % decoder.vocab_size))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let run = |template: &CudaSharedRoutedModelTemplate, first_block: u32| -> Result<Vec<bf16>> {
        let caches = template.allocate_shared_kv()?;
        let mut session = template.instantiate_with_caches(&caches)?;
        let mut table = BlockTable::with_block_size(16);
        table.push(BlockId(first_block));
        table.push(BlockId(first_block + 1));
        table.set_token_len(tokens.len());
        let logits = session.prefill(Uuid::nil(), &tokens, &table)?;
        read(&backend, logits)
    };
    let small = run(&template, 0)?;
    let large = template.with_cache_blocks(8);
    assert_eq!(large.cache_config().block_count, 8);
    let pages = |template: &CudaSharedRoutedModelTemplate| -> Result<usize> {
        Ok(template.allocate_shared_kv()?.iter().flatten().map(PagedKvCache::bytes).sum())
    };
    assert_eq!(pages(&large)?, pages(&template)? * 4);
    // The larger cache places the sequence in blocks the small one lacks.
    assert_eq!(run(&large, 6)?, small);
    Ok(())
}

/// A decode batch that ran over the small cache follows a resize: it binds
/// the sessions' new pages, its paging geometry matches the larger cache,
/// and one step matches the single-session decode.
#[test]
fn decode_batch_follows_a_resized_cache() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut decoder = decoder()?;
    decoder.head_dim = 64;
    decoder.full_attention_partial_rotary_factor = Some(0.09375);
    let fixture = fixture::HybridFixture::nonzero_routed(&decoder)?;
    let cache = CacheConfig {
        block_size: 16,
        block_count: 2,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig { cache, max_sequence_blocks: 2 },
    )?;
    let tokens = (0..8)
        .map(|index| u32::try_from(index % decoder.vocab_size))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut batch = template.prepare_decode_batch(1)?;
    {
        // One step over the small cache binds its pages and captures a graph.
        let caches = template.allocate_shared_kv()?;
        let mut session = template.instantiate_with_caches(&caches)?;
        let mut table = BlockTable::with_block_size(16);
        table.push(BlockId(1));
        table.set_token_len(tokens.len());
        session.prefill(Uuid::from_u128(3), &tokens, &table)?;
        table.set_token_len(tokens.len() + 1);
        let sequence = runtime::backend::DecodeSequence {
            session_id: Uuid::from_u128(3),
            token_id: 3,
            block_table: table,
            sampling_logits: runtime::backend::SamplingLogits::Full,
        };
        batch.execute(&mut [&mut session], std::slice::from_ref(&sequence))?;
    }
    let large = template.with_cache_blocks(8);
    batch.follow_cache_blocks(8);
    let caches = large.allocate_shared_kv()?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(6));
    table.set_token_len(tokens.len());
    let mut batched = large.instantiate_with_caches(&caches)?;
    batched.prefill(Uuid::from_u128(1), &tokens, &table)?;
    let mut single = large.instantiate_with_caches(&caches)?;
    single.prefill(Uuid::from_u128(2), &tokens, &table)?;
    table.set_token_len(tokens.len() + 1);
    let sequence = runtime::backend::DecodeSequence {
        session_id: Uuid::from_u128(1),
        token_id: 3,
        block_table: table.clone(),
        sampling_logits: runtime::backend::SamplingLogits::Full,
    };
    // Full logits are copied into the session rather than sampled on device.
    assert!(batch.execute(&mut [&mut batched], std::slice::from_ref(&sequence))?.is_none());
    let expected = read(&backend, single.decode(Uuid::from_u128(2), 3, &table)?)?;
    let actual = read(&backend, batched.logits())?;
    assert_eq!(actual.len(), expected.len());
    // Batched and single-session kernels round differently in BF16.
    let difference = actual
        .iter()
        .zip(&expected)
        .map(|(left, right)| (f32::from(*left) - f32::from(*right)).abs())
        .fold(0.0_f32, f32::max);
    assert!(difference < 0.01, "logits differ by {difference}");
    Ok(())
}
