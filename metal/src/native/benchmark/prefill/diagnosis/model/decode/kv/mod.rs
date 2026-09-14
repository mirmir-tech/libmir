mod model;

use super::*;
use crate::engine::probe::{Capture, ProjectionKind, ProjectionPair};

#[test]
#[ignore = "actual C3/2049 K/V MXFP4 fusion replay and timing gate; set MIRMIR_BENCH_MODEL"]
fn compares_mxfp4_key_value() -> Result<()> {
    diagnose(2049, &[3, 19], Replay::PerLayer)
}

#[test]
#[ignore = "actual C3/129 K/V dense or MXFP4 fusion numerical gate; set MIRMIR_BENCH_MODEL"]
fn preserves_key_value_projection_join() -> Result<()> {
    diagnose(129, &[3, 19], Replay::Numerical)
}

#[test]
#[ignore = "actual Qwen C3/2049 K/V replay across all attention weights; set MIRMIR_BENCH_MODEL"]
fn compares_interleaved_key_value() -> Result<()> {
    diagnose(2049, &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39], Replay::Interleaved)
}

#[derive(Clone, Copy)]
enum Replay {
    Numerical,
    PerLayer,
    Interleaved,
}

fn diagnose(context: usize, layers: &[usize], replay: Replay) -> Result<()> {
    let mut model = tiles::load_model()?;
    let reference = run_width_with_decode(&mut model, MoePrefill::Default, context, 3, plain)?;
    let mut pairs = Vec::new();
    let targets = layers
        .iter()
        .map(|&layer| (layer, ProjectionKind::KeyValue))
        .collect::<Vec<_>>();
    let observed =
        run_width_with_decode(&mut model, MoePrefill::Default, context, 3, |model, inputs| {
            let started = Instant::now();
            let mut tokens = vec![Vec::new(); inputs.len()];
            for step in 0..32 {
                for (row, input) in inputs.iter().enumerate() {
                    tokens[row].push(input.token);
                }
                let guard = (step == 8).then(|| Capture::begin(&targets)).transpose()?;
                let outputs = model.decode_batch(inputs)?;
                if let Some(guard) = guard {
                    pairs.extend(guard.finish()?.pairs);
                }
                for (input, output) in inputs.iter_mut().zip(outputs) {
                    input.token = greedy_token(&output)?;
                }
            }
            model.stream.synchronize()?;
            Ok(Observation {
                tokens,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            })
        })?;
    assert_eq!(reference.tokens, observed.tokens, "capture changed baseline tokens");
    assert_eq!(reference.schedule, observed.schedule, "capture changed prefill schedule");
    assert_eq!(pairs.iter().map(|pair| pair.layer).collect::<Vec<_>>(), layers);
    writeln!(
        std::io::stderr().lock(),
        "kv.capture: {}",
        json!({
            "context": context, "layers": layers, "observed": observed, "matches_reference": true,
        })
    )?;
    // Both real-model trajectories finish before independent captured replay.
    for pair in &pairs {
        pair.verify(&model.stream)?;
    }
    match replay {
        Replay::Numerical => {},
        Replay::PerLayer => {
            for pair in &pairs {
                pair.qualify(&model.stream)?;
            }
        },
        Replay::Interleaved => ProjectionPair::qualify_interleaved(&pairs, &model.stream)?,
    }
    Ok(())
}
