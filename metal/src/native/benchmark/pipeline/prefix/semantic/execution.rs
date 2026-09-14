use std::io::Write;

use super::*;
use crate::native::model::DecodeExecution;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Schedule {
    Initial,
    Cancelled,
    Refilled,
}

pub(super) fn scalar(
    model: &mut LoadedModel,
    case: usize,
    prompt: &[u32],
    stops: &[u32],
) -> Result<Answer> {
    let (mut input, hit) = prefill(model, prompt, SamplingLogits::Full)?;
    assert_eq!(hit, 0, "scalar reference must start cold");
    let mut tokens = Vec::new();
    let finish = loop {
        if stops.contains(&input.token) {
            break Finish::Eos;
        }
        if tokens.len() == prompts::LIMIT {
            break Finish::Limit;
        }
        tokens.push(input.token);
        let output = model.decode(input.session, input.token, SamplingLogits::Full)?;
        input.token = super::super::super::output_token(model, output)?;
    };
    model.release_session(input.session)?;
    Ok(Answer { case, tokens, finish })
}

pub(super) fn mixed(
    model: &mut LoadedModel,
    prompts: &[Vec<u32>],
    stops: &[u32],
    context: usize,
) -> Result<(Vec<Answer>, usize, usize, Vec<usize>)> {
    let mut rows = Vec::new();
    let mut hits = Vec::new();
    for (case, prompt) in prompts.iter().enumerate() {
        rows.push(start(model, case, prompt, &mut hits)?);
    }
    let mut answers = Vec::new();
    let mut packed = 0;
    let mut full = 0;
    let mut schedule = Schedule::Initial;
    for step in 0..=prompts::LIMIT + prompts::REFILL_STEP {
        if step == prompts::CANCEL_STEP {
            let index = rows.iter().position(|row| row.case == 1).ok_or_else(|| {
                crate::native::error::Error::Benchmark(
                    "cancel target ended before scheduled cancellation".into(),
                )
            })?;
            let row = rows.remove(index);
            model.release_session(row.input.session)?;
            answers.push(Answer {
                case: row.case,
                tokens: row.tokens,
                finish: Finish::Cancelled,
            });
            assert_eq!(schedule, Schedule::Initial);
            schedule = Schedule::Cancelled;
        }
        if step == prompts::REFILL_STEP {
            rows.push(start(model, 1, &prompts[1], &mut hits)?);
            rows.rotate_left(1);
            assert_eq!(schedule, Schedule::Cancelled);
            schedule = Schedule::Refilled;
        }
        let mut pending = Vec::new();
        for mut row in rows {
            let finish = if stops.contains(&row.input.token) {
                Some(Finish::Eos)
            } else if row.tokens.len() == prompts::LIMIT {
                Some(Finish::Limit)
            } else {
                None
            };
            if let Some(finish) = finish {
                model.release_session(row.input.session)?;
                answers.push(Answer {
                    case: row.case,
                    tokens: row.tokens,
                    finish,
                });
            } else {
                row.tokens.push(row.input.token);
                pending.push(row);
            }
        }
        if pending.is_empty() {
            rows = pending;
            if schedule == Schedule::Refilled {
                break;
            }
            continue;
        }
        for (index, row) in pending.iter_mut().enumerate() {
            row.input.sampling = if step % 2 == 0 && index == 0 {
                full += 1;
                SamplingLogits::Full
            } else {
                SamplingLogits::None
            };
        }
        let outputs =
            model.decode_grouped(&pending.iter().map(|row| row.input).collect::<Vec<_>>())?;
        assert_eq!(outputs.len(), pending.len());
        for (row, (output, execution)) in pending.iter_mut().zip(outputs) {
            row.input.token = super::super::super::output_token(model, output)?;
            packed += usize::from(matches!(execution, DecodeExecution::Packed { .. }));
        }
        if step % 3 == 2 {
            pending.rotate_left(1);
        }
        rows = pending;
    }
    assert!(rows.is_empty(), "mixed cohort exceeded bounded schedule");
    assert_eq!(schedule, Schedule::Refilled);
    assert!(packed > 0 && full > 0);
    assert!(
        hits.iter().all(|&hit| hit >= context - 16),
        "shared prefix was not reused: {hits:?}"
    );
    Ok((answers, packed, full, hits))
}

fn start(
    model: &mut LoadedModel,
    case: usize,
    prompt: &[u32],
    hits: &mut Vec<usize>,
) -> Result<Row> {
    let (input, hit) = prefill(model, prompt, SamplingLogits::None)?;
    hits.push(hit);
    writeln!(std::io::stderr().lock(), "prefix.quality.start: case={case}, hit={hit}")?;
    Ok(Row { case, input, tokens: Vec::new() })
}
