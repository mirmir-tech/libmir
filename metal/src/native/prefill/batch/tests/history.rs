use super::{
    fixture::{Family, load_family},
    *,
};
use crate::{
    config::HistoryBatching,
    engine::{MemoryStats, persistent_history::budget::Admission},
    native::model::DecodeInput,
};

fn token(model: &LoadedModel, output: NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(token),
        NativeOutput::Logits(logits) => Ok(logits.argmax_u32(model.stream())?),
    }
}

#[test]
fn pressure_releases_optional_history_and_preserves_nonempty_prefix_cache() -> Result<()> {
    for family in [Family::Hybrid, Family::Clamped] {
        let (mut model, directory) = load_family(family, 10)?;
        let mut inputs = Vec::new();
        let sentinel = (0..32).collect::<Vec<u32>>();
        for seed in [0, 10, 20] {
            let session = Uuid::new_v4();
            let prompt = (0..32).map(|i| (i + seed) % 64).collect::<Vec<_>>();
            let output =
                model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
            let next = token(&model, output.output)?;
            if seed == 0 {
                model.release_session(session)?;
            } else {
                inputs.push(DecodeInput {
                    session,
                    token: next,
                    sampling: SamplingLogits::None,
                });
            }
        }
        model.stream.set_history_batching(HistoryBatching::Persistent);
        model.stream.history_budget().set_admission(Admission::Open);
        for _ in 0..2 {
            let outputs = model.decode_batch(&inputs)?;
            for (input, output) in inputs.iter_mut().zip(outputs) {
                input.token = token(&model, output)?;
            }
        }
        model.stream.synchronize()?;
        let retained = model.stream.history_budget().snapshot().retained_bytes;
        assert_eq!(retained > 0, matches!(family, Family::Hybrid));
        let groups = model.prefixes.group_count();
        assert!(groups > 0, "prefix preservation must not be vacuous");
        model.apply_history_pressure(MemoryStats {
            active: 60,
            cached: 0,
            peak: 0,
            limit: 100,
            recommended: Some(100),
        })?;
        assert_eq!(model.stream.history_budget().snapshot().retained_bytes, 0);
        assert_eq!(model.stream.history_budget().snapshot().admission, Admission::Suspended);
        assert_eq!(model.prefixes.group_count(), groups);
        assert!(model.prefixes.lease_longest(&model.info.manifest.id, &sentinel)?.is_some());
        for input in inputs {
            model.release_session(input.session)?;
        }
        model.clear_prefix_cache();
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
