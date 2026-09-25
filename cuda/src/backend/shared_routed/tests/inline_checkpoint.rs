use runtime::backend::SamplingLogits;

use super::*;

const PROMPT: usize = 40;
const BEFORE: usize = 32;
const BLOCKS: u32 = 4;

/// A checkpoint staged inside one prefill pass must continue like the
/// checkpoint of a pass that ended there, alone and beside other rows.
#[test]
fn staged_checkpoint_continues_like_a_split_prefill() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut decoder = decoder()?;
    decoder.head_dim = 64;
    decoder.full_attention_partial_rotary_factor = Some(0.09375);
    let fixture = fixture::HybridFixture::nonzero_routed(&decoder)?;
    let cache = CacheConfig {
        block_size: 16,
        block_count: BLOCKS * 2,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig {
            cache,
            max_sequence_blocks: BLOCKS as usize,
        },
    )?;
    let tokens = (0..PROMPT)
        .map(|index| u32::try_from(index % decoder.vocab_size))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut tables = (0..2)
        .map(|row| {
            let mut table = BlockTable::with_block_size(16);
            for block in 0..BLOCKS {
                table.push(BlockId(row * BLOCKS + block));
            }
            table
        })
        .collect::<Vec<_>>();

    let caches = template.allocate_shared_kv()?;
    let mut split = template.instantiate_with_caches(&caches)?;
    tables[0].set_token_len(BEFORE);
    split.prefill(Uuid::nil(), &tokens[..BEFORE], &tables[0])?;
    let reference = split.checkpoint()?;
    tables[0].set_token_len(PROMPT);
    split.prefill(Uuid::nil(), &tokens[BEFORE..], &tables[0])?;
    let expected = continuation(&backend, &mut split, &reference, &mut tables[0])?;
    assert!(expected.iter().any(|value| value.to_f32().abs() > 0.01));

    let caches = template.allocate_shared_kv()?;
    let mut single = template.instantiate_with_caches(&caches)?;
    single.arm_checkpoint(BEFORE);
    tables[0].set_token_len(PROMPT);
    single.prefill(Uuid::nil(), &tokens, &tables[0])?;
    let staged = single
        .take_staged_checkpoint(BEFORE)?
        .ok_or(Error::InvalidExecutionPlan("the pass staged no checkpoint"))?;
    assert!(single.take_staged_checkpoint(BEFORE)?.is_none());
    compare(&expected, &continuation(&backend, &mut single, &staged, &mut tables[0])?);

    let caches = template.allocate_shared_kv()?;
    let mut rows = (0..2)
        .map(|_| template.instantiate_with_caches(&caches))
        .collect::<Result<Vec<_>>>()?;
    rows[0].arm_checkpoint(BEFORE);
    let counts = [PROMPT, 7];
    tables[0].set_token_len(PROMPT);
    tables[1].set_token_len(7);
    let mut batch = template.prepare_ragged_prefill_batch(&counts)?;
    batch.execute(
        &mut rows.iter_mut().collect::<Vec<_>>(),
        &[&tokens[..], &tokens[..7]].concat(),
        &tables.iter().collect::<Vec<_>>(),
        &[0, 0],
        Some(&[const { SamplingLogits::None }; 2]),
    )?;
    assert!(rows[1].take_staged_checkpoint(0)?.is_none());
    let staged = rows[0]
        .take_staged_checkpoint(BEFORE)?
        .ok_or(Error::InvalidExecutionPlan("the pass staged no checkpoint"))?;
    compare(&expected, &continuation(&backend, &mut rows[0], &staged, &mut tables[0])?);
    Ok(())
}

/// Logits of one decode after the prompt, then of one decode after restoring
/// `checkpoint`.
fn continuation(
    backend: &CudaBackend,
    session: &mut CudaSharedRoutedModelSession,
    checkpoint: &SharedRoutedCheckpoint,
    table: &mut BlockTable,
) -> Result<Vec<bf16>> {
    assert_eq!(session.position(), PROMPT);
    table.set_token_len(PROMPT + 1);
    let mut logits = read(backend, session.decode(Uuid::nil(), 5, table)?)?;
    session.restore_checkpoint(checkpoint)?;
    assert_eq!(session.position(), BEFORE);
    table.set_token_len(BEFORE + 1);
    logits.extend(read(backend, session.decode(Uuid::nil(), 9, table)?)?);
    Ok(logits)
}

fn compare(expected: &[bf16], actual: &[bf16]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual.to_f32() - expected.to_f32()).abs() < 0.03,
            "logit {index}: {actual:?} != {expected:?}"
        );
    }
}
