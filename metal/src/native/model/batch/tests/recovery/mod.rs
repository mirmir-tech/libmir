use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::*;
use crate::{engine::DecoderModel, native::model::LoadedExecution};

mod execution;

#[test]
fn partial_packed_failure_retires_rows_and_preserves_other_sessions() -> Result<()> {
    exercise(Case::Packed)
}

#[test]
fn partial_scalar_failure_retires_row_and_preserves_other_sessions() -> Result<()> {
    exercise(Case::Scalar)
}

#[test]
fn failed_drain_keeps_resources_and_blocks_execution_until_recovered() -> Result<()> {
    exercise(Case::Drain)
}

#[test]
fn mixed_failure_retires_the_row_whose_output_was_already_produced() -> Result<()> {
    exercise(Case::Mixed)
}

#[test]
fn real_decode_failure_after_tuning_is_not_retried() -> Result<()> {
    exercise(Case::Tuned)
}

#[test]
fn failed_tuning_snapshot_keeps_ownership_until_drain_and_does_not_execute_live_state() -> Result<()>
{
    exercise(Case::Snapshot)
}

#[derive(Clone, Copy)]
enum Case {
    Scalar,
    Packed,
    Mixed,
    Drain,
    Tuned,
    Snapshot,
}

fn config(case: Case) -> crate::MetalConfig {
    let mut config = crate::MetalConfig::default();
    config.cache.paged_attention_min_context = 1;
    config.cache.prefix_cache_entries = 0;
    config.tuning.mode = if matches!(case, Case::Tuned | Case::Snapshot) {
        runtime::tuning::TuningMode::Startup
    } else {
        runtime::tuning::TuningMode::Disabled
    };
    config.tuning.warmup_iterations = 0;
    config.tuning.measurement_iterations = 5;
    config
}

fn exercise(case: Case) -> Result<()> {
    let packed = matches!(case, Case::Packed | Case::Mixed | Case::Drain);
    let fail_drain = matches!(case, Case::Drain | Case::Snapshot);
    let (mut model, directory) = fixture::load_config(config(case))?;
    let (first, first_reference) = pair(&mut model, 0)?;
    let (second, second_reference) = pair(&mut model, 1)?;
    let (survivor, survivor_reference) = pair(&mut model, 2)?;
    let mut inputs = if packed {
        vec![first, second]
    } else {
        vec![first]
    };
    if matches!(case, Case::Mixed) {
        inputs[0].sampling = SamplingLogits::Full;
    }
    let shared_session = Uuid::new_v4();
    let shared = model.sessions[&first.session].snapshot()?;
    model.sessions.insert(shared_session, shared);
    let calls = Arc::new(AtomicUsize::new(0));
    let swapped = std::mem::replace(
        &mut model.execution,
        LoadedExecution::Generation(DecoderModel::new(execution::Empty)),
    );
    let LoadedExecution::Generation(inner) = swapped else {
        unreachable!()
    };
    model.execution = LoadedExecution::Generation(DecoderModel::new(execution::FailAfterWrite {
        inner,
        calls: Arc::clone(&calls),
        timing: match case {
            Case::Tuned => execution::FaultTiming::AfterTuning,
            Case::Snapshot => execution::FaultTiming::DuringTuning,
            _ => execution::FaultTiming::NextForward,
        },
    }));
    if fail_drain {
        crate::native::model::recovery::fail_next_drain();
    }
    let failed = if packed {
        model.decode_grouped(&inputs).map(|_| ())
    } else {
        model.decode(first.session, first.token, first.sampling).map(|_| ())
    };
    assert!(failed.is_err());
    assert_eq!(
        calls.load(Ordering::Relaxed),
        if matches!(case, Case::Tuned) {
            12
        } else {
            1
        }
    );
    if fail_drain {
        let retired = inputs.len() + usize::from(matches!(case, Case::Snapshot));
        assert!(matches!(
            failed,
            Err(crate::native::error::Error::ExecutionRecoveryFailed { .. })
        ));
        assert!(
            matches!(&model.recovery, crate::native::model::recovery::ExecutionRecovery::NeedsDrain(states) if states.len() == retired)
        );
        assert!(model.decode(survivor.session, survivor.token, survivor.sampling).is_err());
        assert!(
            model
                .prefill(Uuid::new_v4(), &[1, 2], &[], SamplingLogits::Full, None, &mut |_| {})
                .is_err()
        );
        model.recover_execution()?;
    }
    for input in &inputs {
        assert!(!model.sessions.contains_key(&input.session), "partial decode remained reusable");
        assert!(model.decode(input.session, input.token, input.sampling).is_err());
        model.release_session(input.session)?;
    }
    let shared = DecodeInput { session: shared_session, ..first };
    for (mut input, reference) in [(survivor, survivor_reference), (shared, first_reference)] {
        for _ in 0..3 {
            let expected = model.decode(reference, input.token, SamplingLogits::Full)?;
            let expected = choose(&model, expected, SamplingLogits::Full)?;
            let actual = model.decode(input.session, input.token, input.sampling)?;
            input.token = choose(&model, actual, input.sampling)?;
            assert_eq!(input.token, expected);
        }
    }
    for session in [
        second.session,
        first_reference,
        second_reference,
        survivor.session,
        survivor_reference,
        shared_session,
    ] {
        model.release_session(session)?;
    }
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
