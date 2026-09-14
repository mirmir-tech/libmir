mod gate;

use gate::{MAX_ROUNDS, Timing, WarmupStatus, warmup_status};

use super::*;

const PLANS: [MoePrefill; 2] = [MoePrefill::Default, MoePrefill::Tiles64];

#[derive(Clone, Copy, serde::Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
enum Phase {
    Warmup { round: usize, slot: usize },
    Measured { block: usize, slot: usize },
}

#[test]
#[ignore = "bounded stable warmup then one C5/2049 BN64 qualification; set MIRMIR_BENCH_MODEL"]
fn compares_settled_compact_tiles() -> Result<()> {
    let mut model = load_model()?;
    let mut reference = None;
    let mut rounds = Vec::new();
    for round in 0..MAX_ROUNDS {
        // Alternate AB and BA so neither plan always starts a warmup round.
        let order = if round % 2 == 0 {
            [0, 1]
        } else {
            [1, 0]
        };
        let first =
            observe(&mut model, PLANS[order[0]], Phase::Warmup { round, slot: 0 }, &mut reference)?;
        let second =
            observe(&mut model, PLANS[order[1]], Phase::Warmup { round, slot: 1 }, &mut reference)?;
        rounds.push(if order[0] == 0 {
            [first, second]
        } else {
            [second, first]
        });
        let status = warmup_status(&rounds);
        writeln!(
            std::io::stderr().lock(),
            "moe.settled_gate: {}",
            json!({
                "rounds": rounds.len(), "status": status,
            })
        )?;
        match status {
            WarmupStatus::Ready => break,
            WarmupStatus::Exhausted => {
                assert_eq!(
                    status,
                    WarmupStatus::Ready,
                    "bounded warmup exhausted; skip measured qualification"
                );
            },
            WarmupStatus::Warming => {},
        }
    }
    assert_eq!(warmup_status(&rounds), WarmupStatus::Ready);
    let mut samples = Vec::new();
    for block in 0..3 {
        let order = if block % 2 == 0 {
            [0, 1, 1, 0]
        } else {
            [1, 0, 0, 1]
        };
        for (slot, index) in order.into_iter().enumerate() {
            let mode = PLANS[index];
            let timing =
                observe(&mut model, mode, Phase::Measured { block, slot }, &mut reference)?;
            samples.push((mode, timing.prefill_ms, timing.decode_ms));
            assert!(
                stable(&samples),
                "unstable measured prefill/decode control; abort qualification"
            );
        }
    }
    Ok(())
}

fn observe(
    model: &mut LoadedModel,
    mode: MoePrefill,
    phase: Phase,
    reference: &mut Option<Observation>,
) -> Result<Timing> {
    model.stream.synchronize()?;
    let before = crate::engine::memory_stats()?;
    let observation = run(model, mode, 2049)?;
    let after_release = crate::engine::memory_stats()?;
    let matches = reference.as_ref().is_none_or(|prior| {
        prior.tokens == observation.tokens && prior.schedule == observation.schedule
    });
    writeln!(
        std::io::stderr().lock(),
        "moe.settled_model: {}",
        json!({
            "phase": phase, "plan": name(mode), "matches_reference": matches,
            "before": allocator(before), "after_release": allocator(after_release),
            "observation": observation,
        })
    )?;
    assert!(matches, "full-model tokens or scheduling changed");
    let timing = Timing {
        prefill_ms: observation.prefill_ms,
        decode_ms: observation.decode_ms,
    };
    if reference.is_none() {
        *reference = Some(observation);
    }
    Ok(timing)
}

fn allocator(memory: crate::engine::MemoryStats) -> serde_json::Value {
    json!({
        "active_bytes": memory.active, "cached_bytes": memory.cached,
        "peak_bytes_process": memory.peak, "limit_bytes": memory.limit,
        "recommended_bytes": memory.recommended,
    })
}
