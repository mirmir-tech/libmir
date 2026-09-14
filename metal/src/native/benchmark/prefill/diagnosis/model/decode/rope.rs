use super::*;
use crate::config::RopeBatching::{self, Offsets, Rows};

#[test]
#[ignore = "ragged C3 offset-array RoPE parity and ABBA; set MIRMIR_BENCH_MODEL"]
fn compares_ragged_rope() -> Result<()> {
    run_modes(&[Rows, Offsets, Rows, Offsets, Offsets, Rows, Offsets, Rows, Rows, Offsets])
}

#[test]
#[ignore = "ragged C3 offset-array RoPE semantic control; set MIRMIR_BENCH_MODEL"]
fn preserves_ragged_rope() -> Result<()> {
    run_modes(&[Rows, Offsets])
}

fn run_modes(order: &[RopeBatching]) -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut reference: Option<super::super::Observation> = None;
    let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
    // Fixed one-run warmup per mode, then two alternating ABBA/BAAB blocks.
    for (run, &plan) in order.iter().enumerate() {
        let observation =
            run_width_with_decode(&mut model, MoePrefill::Default, 2049, 3, |model, inputs| {
                // Existing rows progress 16 tokens before the third joins.
                // Every mode gets identical cold prefill and reference history.
                for _ in 0..16 {
                    let outputs = model.decode_batch(&inputs[..2])?;
                    for (input, output) in inputs[..2].iter_mut().zip(outputs) {
                        input.token = greedy_token(&output)?;
                    }
                }
                model.stream.synchronize()?;
                model.stream.set_rope_batching(plan);
                let result = plain(model, inputs);
                model.stream.set_rope_batching(Rows);
                result
            })?;
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        writeln!(
            std::io::stderr().lock(),
            "decode.rope: {}",
            json!({
                "run": run, "warmup": run < 2, "batched": plan == Offsets,
                "positions": [2065, 2065, 2049], "matches_reference": matches,
                "observation": observation,
            })
        )?;
        assert!(matches, "ragged RoPE changed tokens or schedule");
        if run >= 2 {
            let values = &mut samples[match plan {
                Rows => 0,
                Offsets => 1,
            }];
            values.push(observation.decode_ms);
            let min = values.iter().copied().reduce(f64::min).unwrap_or(0.0);
            let max = values.iter().copied().reduce(f64::max).unwrap_or(0.0);
            assert!(max <= min * 1.15, "RoPE timing spread exceeds 15%; stop");
        }
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    Ok(())
}
