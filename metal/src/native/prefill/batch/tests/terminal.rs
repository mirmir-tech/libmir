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
                    let result = model.prefill(
                        session,
                        &prompt,
                        &checkpoints,
                        &SamplingLogits::Full,
                        None,
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
                        assert_eq!(values.len(), expected.len());
                        for (a, b) in values.iter().zip(expected) {
                            assert!((a - b).abs() <= b.abs().mul_add(1e-4, 1e-4));
                        }
                    } else {
                        reference = Some(values.clone());
                    }
                    model.release_session(session)?;
                    let restored = model.prefill(
                        session,
                        &prompt,
                        &checkpoints,
                        &SamplingLogits::Full,
                        None,
                        &mut |_| {},
                    )?;
                    assert_eq!(restored.prefix_cache_tokens, prompt.len());
                    let NativeOutput::Logits(restored) = restored.output else {
                        return Err(Error::InvalidPrefillBatch("expected cached logits".into()));
                    };
                    assert_eq!(values, restored.to_vec_f32(model.stream())?);
                    model.release_session(session)?;
                }
            }
        }
        model.clear_prefix_cache();
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
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
        model.prefill(session, &prompt, &[16], &SamplingLogits::None, Some(16), &mut |_| {})?;
        model.release_session(session)?;
        prompt[20] = 42;
        let mut expected = None;
        for cached in [true, false] {
            if !cached {
                model.clear_prefix_cache();
            }
            let output = model.prefill(
                session,
                &prompt,
                &[16],
                &SamplingLogits::None,
                Some(16),
                &mut |_| {},
            )?;
            assert_eq!(
                output.prefix_cache_tokens,
                if cached {
                    16
                } else {
                    0
                }
            );
            let mut token = greedy_token(&output.output)?;
            let mut tokens = vec![token];
            for _ in 0..16 {
                token = greedy_token(&model.decode(session, token, SamplingLogits::None)?)?;
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
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

fn greedy_token(output: &NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(*token),
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
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
