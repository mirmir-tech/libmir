use super::*;

#[derive(Clone, Copy)]
enum Case {
    Chunk,
    Completion,
    ExactPrefix,
}

#[test]
fn scalar_prefill_keeps_failed_chunk_until_drain() -> Result<()> {
    exercise(Case::Chunk)
}

#[test]
fn first_token_failure_keeps_state_until_drain() -> Result<()> {
    exercise(Case::Completion)
}

#[test]
fn exact_prefix_failure_preserves_the_reusable_prefix() -> Result<()> {
    exercise(Case::ExactPrefix)
}

fn exercise(case: Case) -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        for batched in [false, true] {
            let (mut model, directory) = load_family(
                family,
                if matches!(case, Case::ExactPrefix) {
                    4
                } else {
                    0
                },
            )?;
            let survivors = support::survivors(&mut model)?;
            let request = request(
                &model,
                if matches!(case, Case::Completion) {
                    1
                } else {
                    32
                },
            );
            if matches!(case, Case::ExactPrefix) {
                model.prefill(
                    request.session_id,
                    &request.prompt_tokens,
                    &[],
                    SamplingLogits::Full,
                    None,
                    &mut |_| {},
                )?;
                model.release_session(request.session_id)?;
            }
            let (stage, fail_at) = match case {
                Case::Chunk => (execution::Stage::Prefill, 1),
                Case::Completion => (execution::Stage::Decode, 2),
                Case::ExactPrefix => (execution::Stage::Decode, 1),
            };
            let calls = execution::install(&mut model, stage, fail_at);
            let batch = if batched {
                Some(
                    MetalPrefillBatch::prepare(
                        &mut model,
                        vec![(request.clone(), SamplingLogits::None)],
                        None,
                    )?
                    .0,
                )
            } else {
                None
            };
            support::arm_drain_failure();
            let result = if let Some(batch) = &batch {
                batch.execute_step(&mut model, 64).map(|_| ())
            } else {
                model
                    .prefill(
                        request.session_id,
                        &request.prompt_tokens,
                        &[],
                        SamplingLogits::None,
                        None,
                        &mut |_| {},
                    )
                    .map(|_| ())
            };
            assert!(matches!(result, Err(Error::ExecutionRecoveryFailed { .. })));
            assert_eq!(calls.load(Ordering::Relaxed), fail_at);
            assert!(!model.sessions.contains_key(&request.session_id));
            if let Some(batch) = batch {
                assert!(batch.finish().is_err());
            }
            support::recover(&mut model, 1, survivors)?;
            // Starting a fresh prefill remains possible; exact hits retain their snapshot.
            let output = model.prefill(
                request.session_id,
                &request.prompt_tokens,
                &[],
                SamplingLogits::Full,
                None,
                &mut |_| {},
            )?;
            if matches!(case, Case::ExactPrefix) {
                assert_eq!(output.prefix_cache_tokens, request.prompt_tokens.len());
            }
            model.release_session(request.session_id)?;
            support::continue_and_release(&mut model, survivors)?;
            drop(model);
            std::fs::remove_dir_all(directory)?;
        }
    }
    Ok(())
}
