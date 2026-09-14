use super::*;

#[test]
#[ignore = "complete Qwen C3/2049 budget256 hardware-gated native prefill/decode; controller required"]
fn qualifies_complete_qwen_budget256() -> std::result::Result<(), DriverError> {
    run_driver(Workload::Complete256)
}

pub(super) fn decode_complete(
    model: &mut LoadedModel,
    inputs: &mut [DecodeInput],
) -> Result<decode::Observation> {
    let started = Instant::now();
    // The first prediction is already produced by prefill and counts in256.
    let mut tokens = inputs.iter().map(|input| vec![input.token]).collect::<Vec<_>>();
    for _ in 1..256 {
        let outputs = model.decode_batch(inputs)?;
        for ((input, output), row) in inputs.iter_mut().zip(outputs).zip(&mut tokens) {
            input.token = greedy_token(&output)?;
            row.push(input.token);
        }
    }
    // Preserve the normal device-token pipeline; drain its final pending graph
    // once at the benchmark boundary, just like the existing shorter sample.
    model.stream.synchronize()?;
    assert!(tokens.iter().all(|row| row.len() == 256));
    Ok(decode::Observation {
        tokens,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}
