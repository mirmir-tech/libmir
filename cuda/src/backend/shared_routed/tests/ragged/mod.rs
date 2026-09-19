mod routed;
mod support;

use runtime::backend::SamplingLogits;
use support::compare_continuation;

use super::*;

#[test]
fn ragged_prefill_and_decode_preserve_nonzero_independent_sessions() -> Result<()> {
    run(false)
}

#[test]
fn ragged_gdn48_preserves_chunked_prefill_and_recurrent_decode() -> Result<()> {
    run(true)
}

#[test]
fn packed_gdn48_preserves_five_checkpoint_tails_and_decode() -> Result<()> {
    run_rounds(true, vec![vec![112; 5], vec![16; 5], vec![1; 5]])
}

#[test]
fn padded_gdn48_preserves_logical_rows_across_token_capacity_reuse() -> Result<()> {
    run_rounds_with_capacity(
        true,
        vec![vec![1008], vec![1008, 1], vec![1, 1008, 1, 1, 1], vec![16; 5], vec![1; 5]],
        Some(1024),
    )
}

#[test]
fn padded_gdn48_reuses_2048_capacity_for_terminal_checkpoint_chunks() -> Result<()> {
    run_rounds_with_capacity(
        true,
        vec![vec![2048], vec![1984], vec![1, 1984, 1], vec![16; 5], vec![1; 5]],
        Some(2048),
    )
}

#[test]
fn padded_hybrid_preserves_small_head_recurrence_and_continuation() -> Result<()> {
    run_rounds_with_capacity(false, vec![vec![17], vec![16, 1], vec![1, 16, 1]], Some(64))
}

fn run(gdn48: bool) -> Result<()> {
    let rounds = if gdn48 {
        vec![vec![960, 1, 63], vec![1024], vec![1, 1023], vec![1, 1, 1], vec![7, 1, 2]]
    } else {
        vec![vec![17, 1, 33], vec![51], vec![1, 50], vec![1, 1, 1], vec![7, 1, 2]]
    };
    run_rounds(gdn48, rounds)
}

fn run_rounds(gdn48: bool, rounds: Vec<Vec<usize>>) -> Result<()> {
    run_rounds_with_capacity(gdn48, rounds, None)
}

fn run_rounds_with_capacity(
    gdn48: bool,
    rounds: Vec<Vec<usize>>,
    capacity: Option<usize>,
) -> Result<()> {
    let rows = rounds
        .iter()
        .map(Vec::len)
        .max()
        .ok_or(Error::InvalidExecutionPlan("test has no prefill rounds"))?;
    let decoder = dense_decoder(gdn48)?;
    let fixture = fixture::HybridFixture::nonzero_dense(&decoder)?;
    run_fixture(&decoder, &fixture, rounds, rows, capacity, true)
}

fn run_fixture(
    decoder: &DecoderConfig,
    fixture: &fixture::HybridFixture,
    rounds: Vec<Vec<usize>>,
    rows: usize,
    capacity: Option<usize>,
    exact_tokens: bool,
) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let blocks = sequence_blocks(&rounds, rows)?;
    let cache = CacheConfig {
        block_size: 16,
        block_count: blocks * u32::try_from(rows)?,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig {
            cache,
            max_sequence_blocks: usize::try_from(blocks)?,
        },
    )?;
    assert!(template.prepare_ragged_prefill_batch(&[]).is_err());
    assert!(template.prepare_ragged_prefill_batch(&[1, 0]).is_err());
    assert!(template.prepare_ragged_prefill_batch(&[usize::MAX, 1]).is_err());
    let caches = template.allocate_shared_kv()?;
    let reference_caches = template.allocate_shared_kv()?;
    let mut actual = (0..rows)
        .map(|_| template.instantiate_with_caches(&caches))
        .collect::<Result<Vec<_>>>()?;
    let mut reference = (0..rows)
        .map(|_| template.instantiate_with_caches(&reference_caches))
        .collect::<Result<Vec<_>>>()?;
    let mut tables = (0..u32::try_from(rows)?)
        .map(|row| {
            let mut table = BlockTable::with_block_size(16);
            for block in 0..blocks {
                table.push(BlockId(row * blocks + block));
            }
            table
        })
        .collect::<Vec<_>>();
    let mut retained = None;
    for counts in rounds {
        let rows = counts.len();
        let starts = actual[..rows]
            .iter()
            .map(CudaSharedRoutedModelSession::position)
            .collect::<Vec<_>>();
        let tokens = counts
            .iter()
            .enumerate()
            .map(|(row, count)| {
                (0..*count)
                    .map(|index| u32::try_from((index + row * 3) % decoder.vocab_size))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut expected = Vec::new();
        for row in 0..rows {
            tables[row].set_token_len(starts[row] + counts[row]);
            reference[row].prefill(
                Uuid::from_u128(u128::try_from(row)?),
                &tokens[row],
                &tables[row],
            )?;
            expected.push(read(&backend, reference[row].sample(SamplingLogits::None)?)?[0]);
        }
        let batch = retained_batch(&mut retained, &template, &counts, capacity)?;
        assert!(
            batch
                .execute(
                    &mut actual[..rows].iter_mut().collect::<Vec<_>>(),
                    &tokens.concat(),
                    &tables[..rows].iter().collect::<Vec<_>>(),
                    &starts,
                    Some(&[]),
                )
                .is_err()
        );
        assert_eq!(
            actual[..rows]
                .iter()
                .map(CudaSharedRoutedModelSession::position)
                .collect::<Vec<_>>(),
            starts
        );
        let selected = batch.execute(
            &mut actual[..rows].iter_mut().collect::<Vec<_>>(),
            &tokens.concat(),
            &tables[..rows].iter().collect::<Vec<_>>(),
            &starts,
            Some(&vec![SamplingLogits::None; rows]),
        )?;
        // Routed logits have near ties whose order follows the tuned kernel
        // choice; the continuation below compares their logits instead.
        assert_eq!(selected.as_ref().map(Vec::len), Some(expected.len()));
        if exact_tokens {
            assert_eq!(selected, Some(expected));
        }
        for row in 0..rows {
            assert_eq!(actual[row].position(), starts[row] + counts[row]);
        }
    }
    compare_continuation(&backend, &mut actual, &mut reference, &mut tables)
}

fn dense_decoder(gdn48: bool) -> Result<DecoderConfig> {
    let mut decoder = decoder()?;
    decoder.num_experts = None;
    decoder.top_k_experts = None;
    decoder.moe_intermediate_size = None;
    decoder.shared_expert_intermediate_size = None;
    decoder.intermediate_size = 64;
    decoder.head_dim = 64;
    decoder.full_attention_partial_rotary_factor = Some(0.09375);
    if gdn48 {
        let linear = decoder
            .linear_attention
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("missing test GDN"))?;
        linear.key_heads = 16;
        linear.value_heads = 48;
        linear.key_head_dim = 128;
        linear.value_head_dim = 128;
    }
    Ok(decoder)
}

fn retained_batch<'a>(
    slot: &'a mut Option<(usize, CudaSharedRoutedPrefillBatch)>,
    template: &CudaSharedRoutedModelTemplate,
    counts: &[usize],
    capacity: Option<usize>,
) -> Result<&'a mut CudaSharedRoutedPrefillBatch> {
    let tokens = capacity.unwrap_or_else(|| counts.iter().sum());
    if slot.as_ref().is_none_or(|(capacity, _)| *capacity != tokens) {
        *slot = Some((tokens, template.prepare_padded_prefill_batch(counts, tokens)?));
    }
    let (_, batch) = slot.as_mut().ok_or(Error::InvalidExecutionPlan("missing test batch"))?;
    assert!(batch.reconfigure(template, &[0]).is_err());
    assert!(batch.reconfigure(template, &[tokens + 1]).is_err());
    batch.reconfigure(template, counts)?;
    Ok(batch)
}

fn sequence_blocks(rounds: &[Vec<usize>], rows: usize) -> Result<u32> {
    Ok(u32::try_from(
        ((0..rows)
            .map(|row| {
                rounds.iter().map(|counts| counts.get(row).copied().unwrap_or(0)).sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            + 16)
            .div_ceil(16),
    )?)
}
