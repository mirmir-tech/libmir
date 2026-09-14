use super::*;
use crate::{
    engine::probe::routing::{Capture, Route},
    native::model::DecodeInput,
};

#[derive(Clone, Copy)]
pub(super) enum Observation {
    Disabled,
    Routes,
}

pub(super) struct Run {
    pub tokens: Vec<Vec<u32>>,
    pub complete: Vec<bool>,
    pub captures: Vec<(usize, Vec<usize>, Vec<Route>)>,
}

pub(super) fn generate(
    model: &mut LoadedModel,
    prompts: &[Vec<u32>],
    stop: &[u32],
    observation: Observation,
) -> Result<Run> {
    model.clear_prefix_cache();
    let mut active = super::super::super::semantic::start(model, prompts)?;
    let mut run = Run {
        tokens: vec![Vec::new(); prompts.len()],
        complete: vec![false; prompts.len()],
        captures: Vec::new(),
    };
    for step in 0..64 {
        let mut pending = Vec::new();
        for (row, input) in active {
            if stop.contains(&input.token) {
                model.release_session(input.session)?;
                run.complete[row] = true;
            } else {
                run.tokens[row].push(input.token);
                pending.push((row, input));
            }
        }
        if pending.is_empty() {
            return Ok(run);
        }
        let inputs: Vec<DecodeInput> = pending.iter().map(|(_, input)| *input).collect();
        let capture = (matches!(observation, Observation::Routes)
            && [0, 1, 2, 4, 8, 16, 24, 32, 48, 63].contains(&step))
        .then(Capture::begin)
        .transpose()?;
        let outputs = if inputs.len() == 1 {
            let input = inputs[0];
            vec![model.decode(input.session, input.token, input.sampling)?]
        } else {
            model.decode_batch(&inputs)?
        };
        if let Some(capture) = capture {
            run.captures.push((
                step,
                pending.iter().map(|(row, _)| *row).collect(),
                capture.finish()?,
            ));
        }
        for ((_, input), output) in pending.iter_mut().zip(outputs) {
            input.token = super::super::super::greedy_token(&output)?;
        }
        active = pending;
    }
    model.stream.synchronize()?;
    for (_, input) in active {
        model.release_session(input.session)?;
    }
    Ok(run)
}
