mod capture;
mod components;
mod gdn;
mod history;
mod kv;
mod moe;
mod probe;
mod rope;
mod width;

use super::*;

pub(super) struct Observation {
    pub tokens: Vec<Vec<u32>>,
    pub elapsed_ms: f64,
}

pub(super) fn plain(model: &mut LoadedModel, inputs: &mut [DecodeInput]) -> Result<Observation> {
    let started = Instant::now();
    let mut tokens = vec![Vec::new(); inputs.len()];
    for _ in 0..32 {
        for (row, input) in inputs.iter().enumerate() {
            tokens[row].push(input.token);
        }
        let outputs = model.decode_batch(inputs)?;
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
        }
    }
    // The device pipeline may still be evaluating the final pending logits.
    // Settle once at the benchmark boundary, preserving overlap between steps.
    model.stream.synchronize()?;
    Ok(Observation {
        tokens,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}
