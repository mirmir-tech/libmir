mod candidate;
mod gather;
mod generation;
mod pages;
mod persistent;
mod reservation;
use super::*;
use crate::engine::probe::history::{Capture, Reader, Replay};

#[test]
#[ignore = "actual Qwen K/V assembly and attention replay; set MIRMIR_BENCH_MODEL"]
fn measures_qwen_history_assembly() -> Result<()> {
    inspect(2049, true)
}

#[test]
#[ignore = "actual GPT-OSS K/V reader inventory; set MIRMIR_BENCH_MODEL"]
fn inspects_clamped_history_readers() -> Result<()> {
    inspect(129, false)
}

fn inspect(context: usize, measure: bool) -> Result<()> {
    let mut model = tiles::load_model()?;
    let reference = run_width_with_decode(&mut model, MoePrefill::Default, context, 3, plain)?;
    let mut records = Vec::new();
    let mut replays = Vec::new();
    let observed =
        run_width_with_decode(&mut model, MoePrefill::Default, context, 3, |model, inputs| {
            let started = Instant::now();
            let mut tokens = vec![Vec::new(); inputs.len()];
            for step in 0..32 {
                for (row, input) in inputs.iter().enumerate() {
                    tokens[row].push(input.token);
                }
                let capture = (step == 31).then(Capture::begin).transpose()?;
                let outputs = model.decode_batch(inputs)?;
                if let Some(capture) = capture {
                    (records, replays) = capture.finish()?;
                }
                for (input, output) in inputs.iter_mut().zip(outputs) {
                    input.token = greedy_token(&output)?;
                }
            }
            // End capture at the final step: no later decode may mutate these page views.
            // Materialize roots before releasing sessions; replay never updates their
            // arena.
            let roots = replays.iter().flat_map(Replay::roots).collect::<Vec<_>>();
            model.stream.eval_many(&roots)?;
            model.stream.synchronize()?;
            Ok(Observation {
                tokens,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            })
        })?;
    assert_eq!(reference.tokens, observed.tokens, "history observer changed tokens");
    assert_eq!(reference.schedule, observed.schedule);
    writeln!(
        std::io::stderr().lock(),
        "history.capture: {}",
        json!({"context":context,
        "records":records,"observed":observed,"matches_reference":true})
    )?;
    if measure {
        assert_eq!(records.len(), 10);
        assert_eq!(replays.len(), 10);
        assert!(records.iter().all(|r| matches!(r.reader, Reader::JoinedView)));
        assert_eq!(
            replays.iter().map(|r| r.layer).collect::<Vec<_>>(),
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        );
    } else {
        assert_eq!(records.len(), 72);
        assert!(replays.is_empty());
        assert!(records.iter().all(|r| matches!(r.reader, Reader::RowView)));
    }
    for replay in &replays {
        replay.inspect(&model.stream)?;
    }
    if measure {
        Replay::measure_assembly(&replays, &model.stream)?;
    }
    Ok(())
}
