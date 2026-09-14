use super::*;

#[test]
#[ignore = "four component-profiled decode steps; set MIRMIR_BENCH_MODEL and RUST_LOG=debug"]
fn profiles_decode_components() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, measure)?;
    writeln!(
        std::io::stderr().lock(),
        "decode.components: {}",
        json!({
            "profiled_steps": [8, 16, 24, 31], "observation": observation,
        })
    )?;
    Ok(())
}

pub(super) fn measure(model: &mut LoadedModel, inputs: &mut [DecodeInput]) -> Result<Observation> {
    let started = Instant::now();
    let mut tokens = vec![Vec::with_capacity(32); inputs.len()];
    for step in 0..32 {
        for (row, input) in inputs.iter().enumerate() {
            tokens[row].push(input.token);
        }
        let profile = [8, 16, 24, 31].contains(&step);
        if profile {
            model.stream.synchronize()?;
        }
        model.stream.set_profile_components(profile);
        let span = tracing::debug_span!("decode_component_step", step).entered();
        let result = model.decode_batch(inputs);
        drop(span);
        model.stream.set_profile_components(false);
        let outputs = result?;
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
        }
    }
    model.stream.synchronize()?;
    Ok(Observation {
        tokens,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}
