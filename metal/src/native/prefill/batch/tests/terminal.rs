use super::{
    fixture::{Family, load_family},
    *,
};
use crate::config::{PrefillMaterialization, PrefillOutput};

#[test]
fn terminal_prefill_preserves_logits_checkpoints_and_exact_hits() -> Result<()> {
    for family in [Family::Hybrid, Family::Clamped] {
        let (mut model, directory) = load_family(family, 8)?;
        model.info.prefill_step = 16;
        for length in [1, 16, 17, 31, 32, 33] {
            let prompt = (0..length).map(|index| index % 64).collect::<Vec<u32>>();
            for checkpoints in [Vec::new(), vec![16, length as usize - 1]] {
                let mut reference: Option<Vec<f32>> = None;
                for (materialization, plan) in
                    [PrefillMaterialization::OutputAndState, PrefillMaterialization::StateOnly]
                        .into_iter()
                        .flat_map(|materialization| {
                            [
                                PrefillOutput::SeparateToken,
                                PrefillOutput::LastChunk,
                                PrefillOutput::DeferredToken,
                            ]
                            .map(|plan| (materialization, plan))
                        })
                {
                    model.stream.set_prefill_materialization(materialization);
                    model.stream.set_prefill_output(plan);
                    model.clear_prefix_cache();
                    let session = Uuid::new_v4();
                    let mut progress = Vec::new();
                    let table = session_table(model.prefix_cache_enabled(), &[0, 1, 2, 3]);
                    let result = model.prefill_with_budget(
                        session,
                        &prompt,
                        &checkpoints,
                        &SamplingLogits::Full,
                        &table,
                        0,
                        &mut |event| progress.push(event.count().current()),
                    )?;
                    if length == 31 && checkpoints.is_empty() {
                        let expected = match plan {
                            PrefillOutput::SeparateToken => vec![0, 16, 30, 31],
                            PrefillOutput::LastChunk | PrefillOutput::DeferredToken => {
                                vec![0, 16, 31]
                            },
                        };
                        assert_eq!(progress, expected);
                    }
                    let NativeOutput::Logits(logits) = result.output else {
                        return Err(Error::InvalidPrefillBatch("expected logits".into()));
                    };
                    assert_eq!(logits.shape()?[..2], [1, 1]);
                    let values = logits.to_vec_f32(model.stream())?;
                    assert!(values.iter().all(|value| value.is_finite()));
                    if let Some(expected) = &reference {
                        if plan != PrefillOutput::LastChunk {
                            assert_eq!(&values, expected);
                        }
                        assert_close(&values, expected);
                    } else {
                        reference = Some(values.clone());
                    }
                    model.release_session(session)?;
                    let cached = usize::from(!model.prefix_cache_enabled()) * prompt.len();
                    let restored = model.prefill_with_budget(
                        session,
                        &prompt,
                        &checkpoints,
                        &SamplingLogits::Full,
                        &table,
                        cached,
                        &mut |_| {},
                    )?;
                    let expected_prefix = if model.prefix_cache_enabled() {
                        prompt.len()
                    } else {
                        runtime::kv::reusable_prefix_tokens(prompt.len(), prompt.len(), Some(16), 0)
                    };
                    assert_eq!(restored.prefix_cache_tokens, expected_prefix);
                    let NativeOutput::Logits(restored) = restored.output else {
                        return Err(Error::InvalidPrefillBatch("expected cached logits".into()));
                    };
                    assert_close(&values, &restored.to_vec_f32(model.stream())?);
                    model.release_session(session)?;
                }
            }
        }
        model.clear_prefix_cache();
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 1);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn terminal_prefill_restores_recurrent_checkpoint_and_continues() -> Result<()> {
    for family in [Family::Hybrid, Family::Clamped] {
        let (mut model, directory) = load_family(family, 8)?;
        model.info.prefill_step = 16;
        model.stream.set_prefill_output(PrefillOutput::LastChunk);
        let mut prompt = (0..31u32).collect::<Vec<_>>();
        let session = Uuid::new_v4();
        let initial = session_table(model.prefix_cache_enabled(), &[0, 1, 2, 3]);
        model.prefill_with_budget(
            session,
            &prompt,
            &[16],
            &SamplingLogits::None,
            &initial,
            0,
            &mut |_| {},
        )?;
        model.release_session(session)?;
        prompt[20] = 42;
        let mut expected = None;
        for cached in [true, false] {
            if !cached {
                model.clear_prefix_cache();
            }
            let ids: &[u32] = if cached {
                &[0, 4, 5, 6]
            } else {
                &[8, 9, 10, 11]
            };
            let table = session_table(model.prefix_cache_enabled(), ids);
            let cached_tokens = usize::from(!model.prefix_cache_enabled() && cached) * 16;
            let output = model.prefill_with_budget(
                session,
                &prompt,
                &[16],
                &SamplingLogits::None,
                &table,
                cached_tokens,
                &mut |_| {},
            )?;
            assert_eq!(output.prefix_cache_tokens, usize::from(cached) * 16);
            let mut token = greedy_token(&output.output)?;
            let mut tokens = vec![token];
            for _ in 0..16 {
                let out = model.decode_with_blocks(
                    session,
                    token,
                    table.blocks(),
                    SamplingLogits::None,
                )?;
                token = greedy_token(&out)?;
                tokens.push(token);
            }
            if let Some(expected) = expected.as_ref() {
                assert_eq!(&tokens, expected);
            } else {
                expected = Some(tokens);
            }
            model.release_session(session)?;
        }
        model.clear_prefix_cache();
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 1);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

fn session_table(prefix_cache: bool, ids: &[u32]) -> BlockTable {
    let mut table = BlockTable::with_block_size(16);
    if !prefix_cache {
        for &id in ids {
            table.push(runtime::kv::BlockId(id));
        }
    }
    table
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= b.abs().mul_add(1e-4, 1e-4));
    }
}

fn greedy_token(output: &NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) | NativeOutput::Verified { next: token, .. } => Ok(*token),
        NativeOutput::Logits(_) => Err(Error::InvalidPrefillBatch("expected greedy output".into())),
    }
}

#[test]
fn terminal_prefill_batch_respects_budget_and_completes_scalar_row() -> Result<()> {
    for family in [Family::Hybrid, Family::Clamped] {
        let (mut model, directory) = load_family(family, 0)?;
        model.info.prefill_step = 16;
        let mut reference = None;
        for (plan, budget) in [
            (PrefillOutput::SeparateToken, 16),
            (PrefillOutput::LastChunk, 16),
            (PrefillOutput::LastChunk, 1),
        ] {
            model.stream.set_prefill_output(plan);
            let request = PrefillRequest {
                model: ModelHandle {
                    id: model.info.manifest.id.clone(),
                    backend: "metal".into(),
                },
                session_id: Uuid::new_v4(),
                prompt_tokens: (0..31).collect(),
                cache_checkpoints: vec![16],
                block_table: BlockTable::with_block_size(16),
                cached_tokens: 0,
                generation_tokens: None,
                sampling_logits: SamplingLogits::None,
            };
            let session = request.session_id;
            let (batch, _) = MetalPrefillBatch::prepare(
                &mut model,
                vec![(request, SamplingLogits::None)],
                None,
            )?;
            let mut prior = 0;
            loop {
                let step = batch.execute_step(&mut model, budget)?;
                let position =
                    step.events.last().map_or(prior, |(_, event)| event.count().current());
                assert!(position > prior && position - prior <= budget as u64);
                prior = position;
                if step.complete {
                    break;
                }
            }
            let output = batch.finish()?.remove(0).native.output;
            let token = greedy_token(&output)?;
            assert_eq!(prior, 31);
            if let Some(expected) = reference {
                assert_eq!(token, expected);
            }
            reference = Some(token);
            model.release_session(session)?;
        }
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 1);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
