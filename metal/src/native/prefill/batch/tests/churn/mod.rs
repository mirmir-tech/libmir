mod diagnosis;
mod width;

use super::{
    fixture::{Family, load_family},
    *,
};
use crate::native::{
    model::{DecodeExecution, DecodeInput},
    prefill::MetalPrefillCohort,
};

#[test]
fn cohort_shrink_refill_reuses_prefixes_and_releases_storage() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 10)?;
        exercise(&mut model)?;
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
#[ignore = "real-model C5/C1/C5 cancellation, prefix reuse and resident storage"]
fn real_model_cohort_shrink_refill_releases_storage()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut model = fixture::load_path(std::env::var("MIRMIR_BENCH_MODEL")?, 10)?;
    exercise(&mut model)?;
    Ok(())
}

fn token(model: &LoadedModel, output: NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(token),
        NativeOutput::Logits(logits) => Ok(logits.argmax_u32(model.stream())?),
    }
}

#[test]
#[ignore = "real-model decode crosses contiguous-to-paged promotion during C5/C1/C5"]
fn real_model_cohort_promotes_contiguous_prefixes()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut model =
        fixture::load_path_with_paging(std::env::var("MIRMIR_BENCH_MODEL")?, 10, 256, 40)?;
    exercise(&mut model)?;
    Ok(())
}

fn prompt(seed: u32) -> Vec<u32> {
    (0..32).map(|index| (seed + index) % 64).collect()
}

fn reference(model: &mut LoadedModel, input: DecodeInput) -> Result<Vec<u32>> {
    let session = Uuid::new_v4();
    let state = model.sessions[&input.session].snapshot()?;
    model.sessions.insert(session, state);
    let mut next = input.token;
    let mut tokens = vec![next];
    for _ in 0..24 {
        let output = model.decode(session, next, SamplingLogits::Full)?;
        next = token(model, output)?;
        tokens.push(next);
    }
    model.release_session(session)?;
    Ok(tokens)
}

fn wave(model: &mut LoadedModel, seeds: &[u32], cached: bool) -> Result<Vec<DecodeInput>> {
    let requests = seeds
        .iter()
        .map(|seed| PrefillRequest {
            model: ModelHandle {
                id: model.info.manifest.id.clone(),
                backend: "metal".into(),
            },
            session_id: Uuid::new_v4(),
            prompt_tokens: prompt(*seed),
            cache_checkpoints: Vec::new(),
            block_table: BlockTable::with_block_size(16),
            cached_tokens: 0,
            generation_tokens: std::num::NonZeroUsize::new(24),
            sampling_logits: SamplingLogits::None,
        })
        .collect::<Vec<_>>();
    let cohort = cached.then(|| MetalPrefillCohort::prepare(model, &requests)).transpose()?;
    let (batch, _) = MetalPrefillBatch::prepare(
        model,
        requests.into_iter().map(|request| (request, SamplingLogits::None)).collect(),
        cohort.as_ref(),
    )?;
    for _ in 0..64 {
        if !batch.execute_step(model, 40)?.complete {
            continue;
        }
        return batch
            .finish()?
            .into_iter()
            .map(|row| {
                assert_eq!(
                    row.native.prefix_cache_tokens,
                    if cached {
                        32
                    } else {
                        0
                    }
                );
                Ok(DecodeInput {
                    session: row.request.session_id,
                    token: token(model, row.native.output)?,
                    sampling: SamplingLogits::None,
                })
            })
            .collect();
    }
    Err(Error::InvalidPrefillBatch("churn prefill made no progress".into()))
}

fn advance(
    model: &mut LoadedModel,
    rows: &mut [(DecodeInput, usize, usize)],
    expected: &[Vec<u32>],
) -> Result<()> {
    let inputs = rows.iter().map(|(input, _, _)| *input).collect::<Vec<_>>();
    let actual = model.decode_grouped(&inputs)?;
    let width = rows.len();
    assert_eq!(actual.len(), width);
    for ((input, seed, offset), (output, execution)) in rows.iter_mut().zip(actual) {
        assert_eq!(
            execution,
            if width == 1 {
                DecodeExecution::Scalar
            } else {
                DecodeExecution::Packed { rows: width }
            }
        );
        *offset += 1;
        input.token = token(model, output)?;
        assert_eq!(input.token, expected[*seed][*offset], "seed {seed}, offset {offset}");
    }
    Ok(())
}

fn recurrent_bytes(model: &LoadedModel, session: Uuid) -> Result<usize> {
    Ok(model.sessions[&session]
        .cache
        .recurrent_allocations()?
        .iter()
        .map(mirtal::memory::Allocation::bytes)
        .sum())
}

fn occupied_pages(model: &LoadedModel) -> Result<usize> {
    let config = model
        .info
        .decoder
        .as_ref()
        .ok_or_else(|| Error::UnsupportedModel("missing decoder".into()))?;
    (0..config.num_hidden_layers)
        .map(|layer| {
            Ok(usize::MAX
                - model.stream().paged_arenas().available_pages(
                    usize::MAX,
                    layer,
                    config.layer_key_value_heads(layer),
                    config.layer_head_dim(layer),
                    crate::engine::KvPageFormat::Native,
                )?)
        })
        .sum()
}

#[allow(clippy::print_stderr)]
pub(super) fn exercise(model: &mut LoadedModel) -> Result<()> {
    for cycle in 0..2 {
        let mut rows = wave(model, &[0, 1, 2, 3, 4], false)?
            .into_iter()
            .enumerate()
            .map(|(seed, input)| (input, seed, 0))
            .collect::<Vec<_>>();
        // Isolate lifecycle behavior from scalar/packed prefill arithmetic.
        model.flush_decode_graphs()?;
        let expected = rows
            .iter()
            .map(|(input, _, _)| reference(model, *input))
            .collect::<Result<Vec<_>>>()?;
        for _ in 0..8 {
            advance(model, &mut rows, &expected)?;
        }
        model.flush_decode_graphs()?;
        let packed_bytes = recurrent_bytes(model, rows[0].0.session)?;
        let full_pages = occupied_pages(model)?;
        assert!(model.stream().paged_arenas().resident_arenas()? > 0);
        for (input, _, _) in rows.drain(1..) {
            model.release_session(input.session)?;
        }
        assert_eq!(model.sessions.len(), 1);
        let survivor_pages = occupied_pages(model)?;
        assert!(survivor_pages < full_pages, "cancelled decode tails still own pages");
        advance(model, &mut rows, &expected)?;
        model.flush_decode_graphs()?;
        let scalar_bytes = recurrent_bytes(model, rows[0].0.session)?;
        if packed_bytes > 0 {
            assert!(
                scalar_bytes <= packed_bytes,
                "survivor recurrence grew after shrinking the cohort"
            );
        }
        for _ in 0..7 {
            advance(model, &mut rows, &expected)?;
        }
        let refill = wave(model, &[1, 2, 3, 4], true)?;
        rows.extend(refill.into_iter().enumerate().map(|(index, input)| {
            assert_eq!(input.token, expected[index + 1][0]);
            (input, index + 1, 0)
        }));
        rows.rotate_left(2);
        for _ in 0..8 {
            advance(model, &mut rows, &expected)?;
        }
        for (input, _, _) in rows {
            model.release_session(input.session)?;
        }
        assert!(model.sessions.is_empty());
        model.clear_prefix_cache();
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0, "cycle {cycle}");
        eprintln!(
            "churn cycle {cycle}: recurrent bytes {packed_bytes} -> {scalar_bytes}, used pages {full_pages} -> {survivor_pages}, all arenas released"
        );
    }
    Ok(())
}
